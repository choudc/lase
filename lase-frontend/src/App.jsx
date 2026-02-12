import { useState, useEffect, useMemo } from 'react'
import { Button } from '@/components/ui/button.jsx'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Textarea } from '@/components/ui/textarea.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs.jsx'
import { ScrollArea } from '@/components/ui/scroll-area.jsx'
import {
  Play,
  Square,
  Settings,
  Plus,
  Clock,
  CheckCircle,
  XCircle,
  AlertCircle,
  Cpu,
  Network,
  Database,
  Trash2,
  MessageSquarePlus,
} from 'lucide-react'
import {
  Panel,
  PanelGroup,
  PanelResizeHandle,
} from "react-resizable-panels"

import {
  createSession as apiCreateSession,
  deleteSession as apiDeleteSession,
  createTask as apiCreateTask,
  getModels as apiGetModels,
  getSessions as apiGetSessions,
  getSessionLogs as apiGetSessionLogs,
  getSessionTasks as apiGetSessionTasks,
  getTools as apiGetTools,
  getUsageSummary as apiGetUsageSummary,
  getOpenAIActualUsage as apiGetOpenAIActualUsage,
  resumeTask as apiResumeTask, // Import new resumeTask API
  submitTaskDecision as apiSubmitTaskDecision,
  pauseTask as apiPauseTask,
  restartTask as apiRestartTask,
  deleteTask as apiDeleteTask,
  refineTask as apiRefineTask,
} from '@/api/lase'
import './App.css'
import { Terminal } from './components/Terminal'
import { Preview } from './components/Preview'
import { SettingsModal } from './components/SettingsModal'
import { TaskOutput } from './components/TaskOutput'
import { IDEPage } from './components/IDEPage'

function App() {
  const inferCategoryFromPrompt = (text) => {
    const t = String(text || '').toLowerCase()
    if (!t.trim()) return null
    if (/(story|novel|narrative|fairy tale|bedtime|plot|character)/.test(t)) return 'story'
    if (/(android app|apk|expo|kotlin|jetpack compose|mobile app)/.test(t)) return 'android_app'
    if (/(python app|python script|flask app|fastapi|django|cli tool)/.test(t)) return 'python_app'
    if (/(website|web app|landing page|frontend|react|vite|html|css)/.test(t)) return 'website'
    if (/(image|illustration|draw|art|picture|photo|poster)/.test(t)) return 'image'
    if (/(research|analyze|analysis|compare|investigate|report|study)/.test(t)) return 'research'
    return null
  }
  const inferStoryDurationFromPrompt = (text) => {
    const t = String(text || '').toLowerCase().trim()
    if (!t) return null
    if (t.includes('half hour') || t.includes('half-hour')) return 30
    const hr = t.match(/\b(\d{1,2})\s*(hours|hour|hrs|hr)\b/)
    if (hr?.[1]) return Math.max(3, Math.min(30, Number(hr[1]) * 60))
    const min = t.match(/\b(\d{1,3})\s*(minutes|minute|mins|min)\b/)
    if (min?.[1]) return Math.max(3, Math.min(30, Number(min[1])))
    const compact = t.match(/\b(\d{1,3})m\b/)
    if (compact?.[1]) return Math.max(3, Math.min(30, Number(compact[1])))
    return null
  }
  const CATEGORY_OPTIONS = [
    { id: 'image', label: 'Image' },
    { id: 'website', label: 'Website' },
    { id: 'research', label: 'Research' },
    { id: 'story', label: 'Story' },
    { id: 'android_app', label: 'Android App' },
    { id: 'python_app', label: 'Python App' },
  ]
  const [sessions, setSessions] = useState([])
  const [currentSession, setCurrentSession] = useState(null)
  const [tasks, setTasks] = useState([])
  const [newTaskDescription, setNewTaskDescription] = useState('')
  const [selectedCategory, setSelectedCategory] = useState('website')
  const [storyGenerateIllustrations, setStoryGenerateIllustrations] = useState(true)
  const [storyIllustrationStyle, setStoryIllustrationStyle] = useState('ghibli')
  const [storyTargetMinutes, setStoryTargetMinutes] = useState(5)
  const [isLoading, setIsLoading] = useState(false)
  const [models, setModels] = useState([])
  const [tools, setTools] = useState([])
  const [logs, setLogs] = useState([])
  const [previewUrl, setPreviewUrl] = useState("http://localhost:3000")
  const [isSettingsOpen, setIsSettingsOpen] = useState(false)
  const [isIDEOpen, setIsIDEOpen] = useState(false)
  const [isResuming, setIsResuming] = useState(false) // New state for resume loading
  const [isPausing, setIsPausing] = useState(false)
  const [isRestarting, setIsRestarting] = useState(false)
  const [usageSummary, setUsageSummary] = useState(null)
  const [actualOpenAIUsage, setActualOpenAIUsage] = useState(null)
  const [taskRefineInputs, setTaskRefineInputs] = useState({})
  const [taskRefineLoading, setTaskRefineLoading] = useState({})
  const [taskDecisionInputs, setTaskDecisionInputs] = useState({})
  const [taskDecisionLoading, setTaskDecisionLoading] = useState({})

  const taskPreviewUrlsById = useMemo(() => {
    const out = {}
    for (const log of logs || []) {
      const taskId = log?.task_id
      if (!taskId) continue
      const data = log?.data || {}
      if (typeof data?.preview_url === 'string' && data.preview_url.trim()) {
        out[taskId] = data.preview_url.trim()
        continue
      }
      const text = typeof data?.output === 'string' ? data.output : ''
      const m = text.match(/Preview URL:\s*(https?:\/\/[^\s]+|\/[^\s]+)/i)
      if (m?.[1]) out[taskId] = m[1]
    }
    return out
  }, [logs])
  const detectedStoryMinutes = useMemo(
    () => inferStoryDurationFromPrompt(newTaskDescription),
    [newTaskDescription]
  )

  useEffect(() => {
    const controller = new AbortController()

      ; (async () => {
        try {
          const [sessionsData, modelsData, toolsData, usageData] = await Promise.all([
            apiGetSessions({ signal: controller.signal }),
            apiGetModels({ signal: controller.signal }),
            apiGetTools({ signal: controller.signal }),
            apiGetUsageSummary({ provider: 'openai', signal: controller.signal }),
          ])

          setSessions(sessionsData)
          setModels(modelsData)
          setTools(toolsData)
          setUsageSummary(usageData)

          if (sessionsData?.length > 0) {
            setCurrentSession((prev) => prev || sessionsData[0])
          }
        } catch (error) {
          if (error?.name !== 'AbortError') console.error('Failed to load initial data:', error)
        }
      })()

    return () => controller.abort()
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    let timerId = null

    const refreshActual = async () => {
      try {
        const data = await apiGetOpenAIActualUsage({ windowDays: 30, signal: controller.signal })
        setActualOpenAIUsage(data)
      } catch (error) {
        if (error?.name === 'AbortError') return
        setActualOpenAIUsage({ error: error?.body?.error || error?.message || 'unknown_error' })
      }
    }

    refreshActual()
    timerId = setInterval(refreshActual, 60000)

    return () => {
      if (timerId) clearInterval(timerId)
      controller.abort()
    }
  }, [])

  useEffect(() => {
    if (currentSession) {
      const controller = new AbortController()

        ; (async () => {
          try {
            const [tasksData, logsData] = await Promise.all([
              apiGetSessionTasks(currentSession.id, { signal: controller.signal }),
              apiGetSessionLogs(currentSession.id, { limit: 50, signal: controller.signal }),
            ])
            setTasks(tasksData)
            setLogs(logsData)
          } catch (error) {
            if (error?.name !== 'AbortError') console.error('Failed to load session data:', error)
          }
        })()

      const intervalId = setInterval(() => {
        // Basic polling for tasks/logs (logs still polled for history, socketio for live)
        Promise.all([
          apiGetSessionTasks(currentSession.id, { signal: controller.signal }),
          apiGetSessionLogs(currentSession.id, { limit: 50, signal: controller.signal }),
          apiGetUsageSummary({ provider: 'openai', signal: controller.signal }),
        ])
          .then(([tasksData, logsData, usageData]) => {
            setTasks(tasksData)
            setLogs(logsData)
            setUsageSummary(usageData)
          })
          .catch((error) => {
            if (error?.name !== 'AbortError') console.error('Failed to refresh session data:', error)
          })
      }, 2000)

      return () => {
        clearInterval(intervalId)
        controller.abort()
      }
    }
  }, [currentSession])

  const createSession = async () => {
    try {
      const newSession = await apiCreateSession({
        name: `Session ${new Date().toLocaleString()}`,
        config: {
          autonomy_level: 'agent',
          network_enabled: false,
        },
      })
      setSessions((prev) => [newSession, ...prev])
      setCurrentSession(newSession)
    } catch (error) {
      console.error('Failed to create session:', error)
    }
  }

  const createTask = async () => {
    if (!newTaskDescription.trim() || !currentSession) return

    setIsLoading(true)
    try {
      const newTask = await apiCreateTask({
        session_id: currentSession.id,
        description: newTaskDescription,
        category: selectedCategory,
        story_options: {
          generate_illustrations: storyGenerateIllustrations,
          illustration_count_mode: 'auto',
          illustration_style: storyIllustrationStyle || 'ghibli',
          duration_minutes: storyTargetMinutes,
        },
        auto_start: true,
      })
      setTasks((prev) => [newTask, ...prev])
      setNewTaskDescription('')
    } catch (error) {
      console.error('Failed to create task:', error)
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    const inferred = inferCategoryFromPrompt(newTaskDescription)
    if (inferred && inferred !== selectedCategory) {
      setSelectedCategory(inferred)
    }
  }, [newTaskDescription, selectedCategory])

  const deleteSession = async (sessionId) => {
    const ok = window.confirm('Delete this session and all its tasks/logs? This cannot be undone.')
    if (!ok) return
    try {
      await apiDeleteSession(sessionId)
      let nextSession = null
      setSessions((prev) => {
        const remaining = prev.filter((s) => s.id !== sessionId)
        nextSession = remaining[0] || null
        return remaining
      })
      if (currentSession?.id === sessionId) {
        setCurrentSession(nextSession)
        if (!nextSession) {
          setTasks([])
          setLogs([])
        }
      }
    } catch (error) {
      const msg = error?.body?.error || error?.message || 'Failed to delete session.'
      window.alert(msg)
      console.error('Failed to delete session:', error)
    }
  }

  const resumeTask = async (taskId) => {
    setIsResuming(true)
    try {
      const resumedTask = await apiResumeTask(taskId)
      setTasks((prev) => prev.map((t) => (t.id === taskId ? resumedTask : t)))
    } catch (error) {
      console.error('Failed to resume task:', error)
    } finally {
      setIsResuming(false)
    }
  }

  const pauseTask = async (taskId) => {
    const ok = window.confirm('Pause this running task?')
    if (!ok) return
    setIsPausing(true)
    try {
      const pausedTask = await apiPauseTask(taskId, { force: true })
      setTasks((prev) => prev.map((t) => (t.id === taskId ? pausedTask : t)))
    } catch (error) {
      const msg = error?.body?.error || error?.message || 'Failed to pause task.'
      window.alert(msg)
      console.error('Failed to pause task:', error)
    } finally {
      setIsPausing(false)
    }
  }

  const restartTask = async (taskId) => {
    const ok = window.confirm('Restart this task from scratch?')
    if (!ok) return
    setIsRestarting(true)
    try {
      const restarted = await apiRestartTask(taskId)
      setTasks((prev) => prev.map((t) => (t.id === taskId ? restarted : t)))
    } catch (error) {
      const msg = error?.body?.error || error?.message || 'Failed to restart task.'
      window.alert(msg)
      console.error('Failed to restart task:', error)
    } finally {
      setIsRestarting(false)
    }
  }

  const deleteTask = async (taskId, taskStatus) => {
    const isRunning = taskStatus === 'running'
    const ok = window.confirm(
      isRunning
        ? 'This task is running. Force stop and delete it permanently?'
        : 'Delete this task permanently? This cannot be undone.'
    )
    if (!ok) return
    try {
      await apiDeleteTask(taskId, { force: isRunning })
      setTasks((prev) => prev.filter((t) => t.id !== taskId))
      setLogs((prev) => prev.filter((l) => l.task_id !== taskId))
    } catch (error) {
      const msg = error?.body?.error || error?.message || 'Failed to delete task.'
      window.alert(msg)
      console.error('Failed to delete task:', error)
    }
  }

  const setTaskRefineInput = (taskId, value) => {
    setTaskRefineInputs((prev) => ({ ...prev, [taskId]: value }))
  }

  const refineTask = async (task) => {
    const msg = (taskRefineInputs[task.id] || '').trim()
    if (!msg) {
      window.alert('Please enter a refinement message first.')
      return
    }
    setTaskRefineLoading((prev) => ({ ...prev, [task.id]: true }))
    try {
      const result = await apiRefineTask(task.id, msg, { autoStart: true })
      const newTask = result?.task || result
      if (newTask?.id) {
        setTasks((prev) => [newTask, ...prev])
      }
      setTaskRefineInputs((prev) => ({ ...prev, [task.id]: '' }))
    } catch (error) {
      const m = error?.body?.error || error?.message || 'Failed to refine task.'
      window.alert(m)
      console.error('Failed to refine task:', error)
    } finally {
      setTaskRefineLoading((prev) => ({ ...prev, [task.id]: false }))
    }
  }

  const setTaskDecisionInput = (taskId, value) => {
    setTaskDecisionInputs((prev) => ({ ...prev, [taskId]: value }))
  }

  const submitDecision = async (task, decision) => {
    const proposal = (taskDecisionInputs[task.id] || '').trim()
    if (decision === 'counter' && !proposal) {
      window.alert('Please provide a counter proposal first.')
      return
    }
    setTaskDecisionLoading((prev) => ({ ...prev, [task.id]: true }))
    try {
      const updated = await apiSubmitTaskDecision(task.id, decision, proposal)
      setTasks((prev) => prev.map((t) => (t.id === task.id ? updated : t)))
      if (decision === 'counter') {
        setTaskDecisionInputs((prev) => ({ ...prev, [task.id]: '' }))
      }
    } catch (error) {
      const msg = error?.body?.error || error?.message || 'Failed to submit decision.'
      window.alert(msg)
    } finally {
      setTaskDecisionLoading((prev) => ({ ...prev, [task.id]: false }))
    }
  }

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed': return 'bg-green-500'
      case 'running': return 'bg-blue-500'
      case 'failed': return 'bg-red-500'
      case 'stopped': return 'bg-yellow-500'
      default: return 'bg-gray-500'
    }
  }

  const openaiUsageLabel = (() => {
    const totals = usageSummary?.totals
    if (!totals) return 'OpenAI(est): 0 req | 0 tok | $0.00'
    const req = Number(totals.requests || 0)
    const tok = Number(totals.total_tokens || 0)
    const usd = Number(totals.cost_usd || 0)
    return `OpenAI(est): ${req} req | ${tok.toLocaleString()} tok | $${usd.toFixed(4)}`
  })()

  const openaiActualUsageLabel = (() => {
    if (!actualOpenAIUsage) return 'OpenAI(actual): loading...'
    if (actualOpenAIUsage?.error) return `OpenAI(actual): ${actualOpenAIUsage.error}`
    const costErr = actualOpenAIUsage?.cost_error
    const usageErr = actualOpenAIUsage?.usage_error
    if (costErr && usageErr) return `OpenAI(actual): unavailable (${costErr}, ${usageErr})`
    const spend = actualOpenAIUsage?.actual_spend_usd
    const req = actualOpenAIUsage?.actual_requests
    const inTok = actualOpenAIUsage?.actual_input_tokens
    const outTok = actualOpenAIUsage?.actual_output_tokens
    const spendLabel = Number.isFinite(Number(spend)) ? `$${Number(spend).toFixed(4)}` : '$n/a'
    const reqLabel = Number.isFinite(Number(req)) ? `${Number(req).toLocaleString()} req` : 'n/a req'
    const tokLabel = Number.isFinite(Number(inTok)) && Number.isFinite(Number(outTok))
      ? `${Number(inTok).toLocaleString()}/${Number(outTok).toLocaleString()} tok`
      : 'n/a tok'
    const days = Number(actualOpenAIUsage?.window_days || 30)
    return `OpenAI(actual ${days}d): ${reqLabel} | ${tokLabel} | ${spendLabel}`
  })()

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed': return <CheckCircle className="h-4 w-4" />
      case 'running': return <Play className="h-4 w-4" />
      case 'failed': return <XCircle className="h-4 w-4" />
      case 'stopped': return <Square className="h-4 w-4" />
      default: return <Clock className="h-4 w-4" />
    }
  }

  const safeProgress = (task) => {
    const p = Number(task?.progress)
    if (!Number.isFinite(p)) return 0
    return Math.min(1, Math.max(0, p))
  }

  const taskToPreviewUrl = (task) => {
    const text = task?.last_output || ''
    if (!text) return null
    const previewMatch = text.match(/Preview URL:\s*(https?:\/\/[^\s]+|\/[^\s]+)/i)
    if (previewMatch?.[1]) return previewMatch[1]
    const webPortMatch = text.match(/Started\s+web\s+on\s+port\s+(\d+)/i)
    if (webPortMatch?.[1]) return `http://localhost:${webPortMatch[1]}`
    const imagePathMatch = text.match(/Image generated successfully:\s*(.+\.(png|jpg|jpeg|webp))/i)
    if (imagePathMatch?.[1]) {
      const fullPath = imagePathMatch[1].trim()
      const filename = fullPath.split(/[\\/]/).pop()
      if (filename) return `/api/images/${filename}`
    }
    return null
  }

  const resolveTaskPreviewUrl = (task) =>
    taskToPreviewUrl(task) || taskPreviewUrlsById[task?.id] || null

  useEffect(() => {
    const latestCompletedStory = (tasks || []).find(
      (t) => String(t?.category || '').toLowerCase() === 'story' && String(t?.status || '') === 'completed'
    )
    if (!latestCompletedStory) return
    const nextUrl = resolveTaskPreviewUrl(latestCompletedStory)
    if (!nextUrl) return
    setPreviewUrl((prev) => (prev === nextUrl ? prev : nextUrl))
  }, [tasks, taskPreviewUrlsById])

  const getTaskOutputType = (task) => {
    const category = String(task?.category || '').toLowerCase()
    if (category === 'image') return 'image'
    if (category === 'website') return 'website'
    if (category === 'story') return 'story'
    if (category === 'android_app' || category === 'python_app') return 'code'

    const text = task?.last_output || ''
    const description = task?.description || ''
    const previewUrl = resolveTaskPreviewUrl(task) || ''

    const isImage =
      task?.status_detail === 'image_generated' ||
      /\/api\/images\//i.test(previewUrl) ||
      /image generated successfully/i.test(text)
    if (isImage) return 'image'

    const isWebsite =
      /^https?:\/\//i.test(previewUrl) ||
      /started\s+web\s+on\s+port\s+\d+/i.test(text)
    if (isWebsite) return 'website'

    const codeLike =
      /```/.test(text) ||
      /(code|implement|refactor|function|class|api|frontend|backend|component|script|bug|fix|file)/i.test(description)
    if (codeLike) return 'code'

    return 'other'
  }

  const getTaskBrief = (task) => {
    const description = (task?.description || '').trim()
    const outputType = getTaskOutputType(task)
    const status = task?.status || 'queued'
    const statusDetail = (task?.status_detail || '').trim()
    const lastOutput = (task?.last_output || '').trim()

    let intent = 'General task'
    if (outputType === 'image') intent = 'Image generation'
    else if (outputType === 'website') intent = 'Website implementation'
    else if (outputType === 'code') intent = 'Code implementation'

    let activity = 'waiting to start'
    if (status === 'running') activity = statusDetail ? statusDetail.replaceAll('_', ' ') : 'in progress'
    else if (status === 'completed') activity = 'completed successfully'
    else if (status === 'failed') activity = statusDetail ? `failed: ${statusDetail.replaceAll('_', ' ')}` : 'failed'
    else if (status === 'stopped') activity = 'stopped'

    if (lastOutput) {
      const firstLine = lastOutput.split('\n').map((l) => l.trim()).find((l) => !!l) || ''
      if (firstLine && !firstLine.startsWith('```')) {
        const normalized = firstLine.length > 90 ? `${firstLine.slice(0, 90)}...` : firstLine
        if (status === 'running') activity = normalized
      }
    }

    if (description.length > 0 && (intent === 'General task' || outputType === 'other')) {
      const short = description.length > 70 ? `${description.slice(0, 70)}...` : description
      return `${short} (${activity})`
    }
    return `${intent}: ${activity}`
  }

  const getTaskModelLabel = (task) => {
    const snap = task?.context_snapshot || {}
    const m = snap?.model_info || snap?.model_used || null
    if (m && typeof m === 'object') {
      const provider = String(m.provider || '').trim()
      const name = String(m.name || '').trim()
      if (provider && name) return `${provider}:${name}`
      if (name) return name
      if (provider) return provider
    }
    return 'pending'
  }

  useEffect(() => {
    if (!tasks?.length) return
    const candidate = tasks.find((t) => ['running', 'completed'].includes(t.status))
    if (!candidate) return
    const nextUrl = resolveTaskPreviewUrl(candidate)
    if (nextUrl && nextUrl !== previewUrl) setPreviewUrl(nextUrl)
  }, [tasks, logs, previewUrl, taskPreviewUrlsById])

  const currentPlanSteps = useMemo(() => {
    const activeTask = tasks?.[0]
    const activeTaskId = activeTask?.id
    if (!activeTaskId) return []
    const taskLogs = (logs || []).filter((l) => l.task_id === activeTaskId)
    let steps = []
    let statuses = []
    let sawTaskCompleted = false
    let sawTaskFailed = false

    for (const log of taskLogs) {
      if (log.event_type === 'plan_initialized') {
        const s = Array.isArray(log?.data?.steps) ? log.data.steps : []
        steps = s.slice(0, 20)
        statuses = steps.map(() => 'pending')
      } else if (log.event_type === 'plan_step_update') {
        const idx = Number(log?.data?.index)
        if (!Number.isFinite(idx) || idx < 0) continue
        while (statuses.length <= idx) {
          statuses.push('pending')
          steps.push(`Step ${statuses.length}`)
        }
        const st = String(log?.data?.status || '')
        if (st === 'completed' || st === 'completed_after_retry') statuses[idx] = 'completed'
        else if (st === 'error') statuses[idx] = 'error'
      } else if (log.event_type === 'task_completed') {
        sawTaskCompleted = true
      } else if (log.event_type === 'task_failed') {
        sawTaskFailed = true
      }
    }

    // Reconcile checklist with terminal task state even if explicit step updates were skipped.
    if (sawTaskCompleted || activeTask?.status === 'completed') {
      statuses = steps.map((_, i) => (statuses[i] === 'error' ? 'error' : 'completed'))
    } else if (sawTaskFailed || activeTask?.status === 'failed') {
      const firstPending = statuses.findIndex((s) => s === 'pending')
      if (firstPending >= 0) statuses[firstPending] = 'error'
    }

    return steps.map((name, i) => ({ name, status: statuses[i] || 'pending' }))
  }, [logs, tasks])

  return (
    <div className="flex h-screen flex-col bg-gray-50 dark:bg-gray-900 overflow-hidden">
      {/* Header */}
      <header className="glass sticky top-0 z-50 px-6 py-3 flex-none">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <Cpu className="h-6 w-6 text-blue-600" />
              <h1 className="text-xl font-bold text-gray-900 dark:text-white">LASE</h1>
            </div>
            <Badge variant="outline">Mission Control</Badge>
          </div>
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2 px-3 py-1 bg-gray-100 dark:bg-gray-700 rounded text-xs font-mono">
              <Network className="h-3 w-3 mr-2" />
              SocketIO: Connected
            </div>
            <div className="flex items-center space-x-2 px-3 py-1 bg-gray-100 dark:bg-gray-700 rounded text-xs font-mono">
              <Database className="h-3 w-3 mr-2" />
              {openaiUsageLabel}
            </div>
            <div className="flex items-center space-x-2 px-3 py-1 bg-gray-100 dark:bg-gray-700 rounded text-xs font-mono">
              <Database className="h-3 w-3 mr-2" />
              {openaiActualUsageLabel}
            </div>
            <Button variant="outline" size="sm" onClick={() => setIsSettingsOpen(true)}>
              <Settings className="h-4 w-4 mr-2" />
              Settings
            </Button>
          </div>
        </div>
      </header>

      <SettingsModal open={isSettingsOpen} onOpenChange={setIsSettingsOpen} />

      {isIDEOpen && currentSession ? (
        <IDEPage session={currentSession} onClose={() => setIsIDEOpen(false)} />
      ) : (
        <div className="flex-1 min-h-0 overflow-hidden">
          <PanelGroup direction="horizontal" className="h-full">

            {/* Left Panel: Sessions */}
            <Panel defaultSize={20} minSize={15} maxSize={30} className="bg-white dark:bg-gray-800 border-r dark:border-gray-700">
              <div className="h-full flex flex-col p-4">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-sm font-semibold text-gray-900 dark:text-white uppercase tracking-wider">Sessions</h2>
                  <Button onClick={createSession} size="sm" variant="ghost" className="h-8 w-8 p-0">
                    <Plus className="h-4 w-4" />
                  </Button>
                </div>
                <ScrollArea className="flex-1 -mx-2 px-2">
                  <div className="space-y-2">
                    {sessions.map((session) => (
                          <Card
                        key={session.id}
                        className={`cursor-pointer transition-all ${currentSession?.id === session.id ? 'ring-1 ring-blue-500 shadow-md' : 'hover:bg-gray-50'
                          }`}
                        onClick={() => setCurrentSession(session)}
                      >
                        <CardContent className="p-3">
                          <div className="flex items-center justify-between mb-1 gap-2">
                            <span className="font-medium text-sm truncate">{session.name}</span>
                            <div className="flex items-center gap-2">
                              <div className={`w-2 h-2 rounded-full ${getStatusColor(session.status)}`} />
                              <button
                                type="button"
                                title="Delete session"
                                className="text-red-500 hover:text-red-700"
                                onClick={(e) => {
                                  e.stopPropagation()
                                  deleteSession(session.id)
                                }}
                              >
                                <Trash2 className="h-3.5 w-3.5" />
                              </button>
                            </div>
                          </div>
                          <p className="text-xs text-gray-500">{session.task_count} tasks</p>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </ScrollArea>
                <div className="mt-3 border-t pt-3">
                  <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-2">Current Plan</div>
                  {currentPlanSteps.length === 0 ? (
                    <div className="text-xs text-gray-400">No plan yet</div>
                  ) : (
                    <div className="space-y-1 max-h-40 overflow-auto pr-1">
                      {currentPlanSteps.map((step, idx) => (
                        <div key={`${idx}-${step.name}`} className="text-xs flex items-start gap-2">
                          <span className={
                            step.status === 'completed'
                              ? 'text-green-600'
                              : step.status === 'error'
                                ? 'text-red-600'
                                : 'text-gray-400'
                          }>
                            {step.status === 'completed' ? '✓' : step.status === 'error' ? '✕' : '○'}
                          </span>
                          <span className="text-gray-600 dark:text-gray-300 leading-4">{step.name}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </Panel>

            <PanelResizeHandle />

            {/* Middle Panel: Task Execution (Agent View) */}
            <Panel defaultSize={45} minSize={30}>
              <div className="h-full min-h-0 flex flex-col bg-white dark:bg-gray-800">
                {/* Task Input */}
                <div className="p-4 border-b dark:border-gray-700">
                  <div className="mb-2 flex flex-wrap gap-2">
                    {CATEGORY_OPTIONS.map((c) => (
                      <Button
                        key={c.id}
                        size="sm"
                        variant={selectedCategory === c.id ? 'default' : 'outline'}
                        className="h-7 text-xs"
                        onClick={() => setSelectedCategory(c.id)}
                      >
                        {c.label}
                      </Button>
                    ))}
                  </div>
                  {selectedCategory === 'story' && (
                    <div className="mb-3 flex flex-wrap items-center gap-3 rounded border border-amber-200 bg-amber-50/60 px-3 py-2 text-xs">
                      <label className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          checked={storyGenerateIllustrations}
                          onChange={(e) => setStoryGenerateIllustrations(e.target.checked)}
                        />
                        Generate illustration images
                      </label>
                      <span className="text-slate-700">Images: Auto (based on story length)</span>
                      <label className="flex items-center gap-2">
                        Style:
                        <select
                          value={storyIllustrationStyle}
                          onChange={(e) => setStoryIllustrationStyle(e.target.value)}
                          className="rounded border bg-white px-2 py-1"
                          disabled={!storyGenerateIllustrations}
                        >
                          <option value="ghibli">Ghibli (default)</option>
                          <option value="storybook">Storybook</option>
                          <option value="anime">Anime</option>
                          <option value="cinematic">Cinematic</option>
                          <option value="fantasy">Fantasy</option>
                          <option value="watercolor">Watercolor</option>
                          <option value="photorealistic">Photorealistic</option>
                        </select>
                      </label>
                      <label className="flex items-center gap-2">
                        Story length:
                        <select
                          value={storyTargetMinutes}
                          onChange={(e) => setStoryTargetMinutes(Number(e.target.value))}
                          className="rounded border bg-white px-2 py-1"
                        >
                          <option value={3}>3 min</option>
                          <option value={5}>5 min</option>
                          <option value={10}>10 min</option>
                          <option value={15}>15 min</option>
                          <option value={30}>30 min</option>
                        </select>
                      </label>
                      <span className="text-slate-700">
                        {detectedStoryMinutes
                          ? `Detected from prompt: ${detectedStoryMinutes} min (overrides selector)`
                          : 'Used when prompt does not specify duration'}
                      </span>
                    </div>
                  )}
                  <div className="flex gap-2">
                    <Textarea
                      placeholder="Instruction (e.g., 'Create a React app called dashboard')"
                      value={newTaskDescription}
                      onChange={(e) => setNewTaskDescription(e.target.value)}
                      className="flex-1 min-h-[60px] resize-none"
                    />
                    <Button
                      onClick={createTask}
                      disabled={!newTaskDescription.trim() || isLoading}
                      className="h-auto w-24 bg-blue-600 hover:bg-blue-700"
                    >
                      {isLoading ? <AlertCircle className="animate-spin" /> : <Play />}
                    </Button>
                  </div>
                </div>

                {/* Tasks List */}
                <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
                  <Tabs defaultValue="tasks" className="flex-1 min-h-0 flex flex-col">
                    <div className="px-4 pt-2 border-b">
                      <TabsList className="w-full justify-start h-9 bg-transparent p-0">
                        <TabsTrigger value="tasks" className="data-[state=active]:border-b-2 border-blue-600 rounded-none px-4">Tasks</TabsTrigger>
                        <TabsTrigger value="logs" className="data-[state=active]:border-b-2 border-blue-600 rounded-none px-4">Agent Logs</TabsTrigger>
                      </TabsList>
                    </div>

                    <TabsContent value="tasks" className="flex-1 min-h-0 p-4 m-0 overflow-auto bg-gray-50/50">
                      <div className="space-y-4">
                        {tasks.map((task) => (
                          <Card key={task.id} className="border-l-4 border-l-blue-500">
                            <CardHeader className="py-3 px-4 pb-2">
                              <div className="flex justify-between items-start">
                                <div className="flex items-center gap-2">
                                  {getStatusIcon(task.status)}
                                  <span className="font-medium text-sm">Task {task.id.slice(0, 6)}</span>
                                  {task.category && (
                                    <Badge variant="outline" className="text-[10px] uppercase">{String(task.category).replaceAll('_', ' ')}</Badge>
                                  )}
                                  <Badge variant="outline" className="text-[10px]">model: {getTaskModelLabel(task)}</Badge>
                                </div>
                                <Badge variant="secondary" className="text-xs">{task.status}</Badge>
                              </div>
                              <p className="text-sm text-gray-600 mt-1">{task.description}</p>
                              <p className="text-xs text-gray-500 mt-1">{getTaskBrief(task)}</p>
                            </CardHeader>
                            <CardContent className="px-4 py-2 pb-3">
                              {(() => {
                                const outputType = getTaskOutputType(task)
                                const taskPreviewUrl = resolveTaskPreviewUrl(task)
                                const showLivePreview = (outputType === 'image' || outputType === 'website' || outputType === 'story') && !!taskPreviewUrl
                                const showIDE = outputType === 'code' || outputType === 'website'
                                if (!showLivePreview && !showIDE) return null
                                return (
                                  <div className="mt-2 flex gap-2">
                                    {showLivePreview && (
                                      <Button
                                        size="sm"
                                        variant="outline"
                                        className="text-xs h-7"
                                        onClick={() => setPreviewUrl(taskPreviewUrl)}
                                      >
                                        Open in Live Preview
                                      </Button>
                                    )}
                                    {showIDE && (
                                      <Button
                                        size="sm"
                                        variant="outline"
                                        className="text-xs h-7"
                                        onClick={() => setIsIDEOpen(true)}
                                      >
                                        Open in IDE
                                      </Button>
                                    )}
                                  </div>
                                )
                              })()}
                              {task.status === 'running' && (
                                <div className="mt-2">
                                  <div className="w-full bg-gray-200 h-1.5 rounded-full overflow-hidden">
                                    <div className="bg-blue-500 h-full animate-pulse" style={{ width: `${safeProgress(task) * 100}%` }} />
                                  </div>
                                  <Button
                                    size="sm"
                                    variant="outline"
                                    onClick={() => pauseTask(task.id)}
                                    disabled={isPausing}
                                    className="w-full text-xs h-7 mt-2"
                                  >
                                    {isPausing ? 'Pausing...' : 'Pause Task'}
                                  </Button>
                                </div>
                              )}
                              {(task.status === 'failed' || task.status === 'stopped') && (
                                <div className="mt-3 space-y-2">
                                  <Button
                                    size="sm"
                                    onClick={() => resumeTask(task.id)}
                                    disabled={isResuming}
                                    className="w-full text-xs h-7 bg-green-600 hover:bg-green-700"
                                  >
                                    {isResuming ? 'Resuming...' : 'Resume Task'}
                                  </Button>
                                  <Button
                                    size="sm"
                                    variant="outline"
                                    onClick={() => restartTask(task.id)}
                                    disabled={isRestarting}
                                    className="w-full text-xs h-7"
                                  >
                                    {isRestarting ? 'Restarting...' : 'Restart Task'}
                                  </Button>
                                </div>
                              )}
                              {task.status === 'completed' && (
                                <div className="mt-2">
                                  <Button
                                    size="sm"
                                    variant="outline"
                                    onClick={() => restartTask(task.id)}
                                    disabled={isRestarting}
                                    className="w-full text-xs h-7"
                                  >
                                    {isRestarting ? 'Restarting...' : 'Restart Task'}
                                  </Button>
                                </div>
                              )}
                              <div className="mt-2">
                                <Button
                                  size="sm"
                                  variant="outline"
                                  onClick={() => deleteTask(task.id, task.status)}
                                  className="w-full text-xs h-7 text-red-600 border-red-300 hover:bg-red-50"
                                >
                                  Delete Task
                                </Button>
                              </div>
                              {task.status_detail === 'awaiting_user_decision' && (
                                <div className="mt-3 border rounded-md bg-amber-50/70 dark:bg-amber-900/10 p-2">
                                  <div className="text-[11px] text-amber-700 dark:text-amber-300 mb-2">
                                    Agent is waiting for your decision
                                  </div>
                                  <div className="flex gap-2 mb-2">
                                    <Button
                                      size="sm"
                                      onClick={() => submitDecision(task, 'accept')}
                                      disabled={Boolean(taskDecisionLoading[task.id])}
                                      className="h-7 text-xs bg-green-600 hover:bg-green-700"
                                    >
                                      Accept
                                    </Button>
                                    <Button
                                      size="sm"
                                      variant="outline"
                                      onClick={() => submitDecision(task, 'counter')}
                                      disabled={Boolean(taskDecisionLoading[task.id])}
                                      className="h-7 text-xs"
                                    >
                                      Submit Counter Proposal
                                    </Button>
                                  </div>
                                  <Textarea
                                    value={taskDecisionInputs[task.id] || ''}
                                    onChange={(e) => setTaskDecisionInput(task.id, e.target.value)}
                                    placeholder="Counter proposal (optional unless you choose counter)..."
                                    className="min-h-[56px] resize-y text-xs"
                                  />
                                </div>
                              )}
                              <div className="mt-3 border rounded-md bg-white/70 dark:bg-gray-900/40 p-2">
                                <div className="text-[11px] text-gray-500 mb-1">Task Chat / Refinement</div>
                                <div className="flex gap-2">
                                  <Textarea
                                    value={taskRefineInputs[task.id] || ''}
                                    onChange={(e) => setTaskRefineInput(task.id, e.target.value)}
                                    placeholder="Add more details, corrections, or refinement instructions for this task..."
                                    className="min-h-[64px] resize-y text-xs"
                                  />
                                  <Button
                                    size="sm"
                                    className="h-auto bg-blue-600 hover:bg-blue-700"
                                    onClick={() => refineTask(task)}
                                    disabled={Boolean(taskRefineLoading[task.id])}
                                    title="Create a refinement follow-up task"
                                  >
                                    <MessageSquarePlus className="h-4 w-4" />
                                  </Button>
                                </div>
                              </div>
                              {task.last_output && (
                                <div className="mt-3 bg-gray-50 dark:bg-gray-900/50 p-4 rounded-lg border dark:border-gray-800 max-h-[45vh] overflow-auto">
                                  <TaskOutput content={task.last_output} />
                                </div>
                              )}
                            </CardContent>
                          </Card>
                        ))}
                      </div>
                    </TabsContent>

                    <TabsContent value="logs" className="flex-1 p-0 m-0 overflow-auto bg-black">
                      <div className="p-4 font-mono text-xs space-y-1">
                        {logs.map(log => (
                          <div key={log.id} className="text-gray-300 border-b border-gray-800 pb-1 mb-1">
                            <span className="text-blue-400">[{new Date(log.timestamp).toLocaleTimeString()}]</span>
                            <span className="text-green-500 font-bold ml-2">{log.event_type}</span>
                            <div className="pl-4 text-gray-400 whitespace-pre-wrap">{JSON.stringify(log.data)}</div>
                          </div>
                        ))}
                      </div>
                    </TabsContent>
                  </Tabs>
                </div>
              </div>
            </Panel>

            <PanelResizeHandle />

            {/* Right Panel: Preview & Terminal */}
            <Panel defaultSize={35} minSize={20}>
              <PanelGroup direction="vertical" className="h-full">
                <Panel defaultSize={50} minSize={20} className="bg-gray-100 dark:bg-gray-900 p-2">
                  <div className="h-full flex flex-col">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-semibold uppercase text-gray-500">Live Preview</span>
                      <input
                        className="text-xs border rounded px-2 py-0.5 w-48 bg-white"
                        value={previewUrl}
                        onChange={(e) => setPreviewUrl(e.target.value)}
                      />
                    </div>
                    <div className="flex-1 overflow-hidden rounded-lg shadow-sm">
                      <Preview url={previewUrl} sessionId={currentSession?.id} />
                    </div>
                  </div>
                </Panel>

                <PanelResizeHandle />

                <Panel defaultSize={50} minSize={20} className="bg-black p-2">
                  <div className="h-full flex flex-col">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-semibold uppercase text-gray-500">Terminal Output (PID)</span>
                    </div>
                    <div className="flex-1 overflow-hidden rounded border border-gray-800">
                      <Terminal />
                    </div>
                  </div>
                </Panel>
              </PanelGroup>
            </Panel>

          </PanelGroup >
        </div>
      )}
    </div>
  )
}

export default App
