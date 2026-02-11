import { requestJson } from './http'

export function getApiBase() {
  // Prefer explicit env, otherwise assume same-origin backend serves /api.
  // In dev, Vite proxy (vite.config.js) forwards /api to the backend.
  return (import.meta.env.VITE_API_BASE || '/api').replace(/\/+$/, '')
}

export function getSessions({ signal } = {}) {
  return requestJson(getApiBase(), '/sessions', { signal })
}

export function createSession({ name, config }, { signal } = {}) {
  return requestJson(getApiBase(), '/sessions', {
    method: 'POST',
    body: { name, config },
    signal,
  })
}

export function deleteSession(sessionId, { signal } = {}) {
  return requestJson(getApiBase(), `/sessions/${sessionId}`, {
    method: 'DELETE',
    signal,
  }).catch((error) => {
    const status = Number(error?.status || error?.body?.status || 0)
    if (status !== 405) throw error
    // Compatibility fallback for older proxies/backends blocking DELETE.
    return requestJson(getApiBase(), `/sessions/${sessionId}/delete`, {
      method: 'POST',
      signal,
    })
  })
}

export function getModels({ signal } = {}) {
  return requestJson(getApiBase(), '/models', { signal })
}

export function getAvailableModels({ signal } = {}) {
  return getModels({ signal });
}

export function getTools({ signal } = {}) {
  return requestJson(getApiBase(), '/tools', { signal })
}

export function getUsageSummary({ provider = 'openai', signal } = {}) {
  return requestJson(getApiBase(), `/usage/summary?provider=${encodeURIComponent(provider)}`, { signal })
}

export function getOpenAIActualUsage({ windowDays = 30, signal } = {}) {
  return requestJson(
    getApiBase(),
    `/usage/openai/actual?window_days=${encodeURIComponent(windowDays)}`,
    { signal }
  )
}

export function getSessionTasks(sessionId, { signal } = {}) {
  return requestJson(getApiBase(), `/sessions/${sessionId}/tasks`, { signal })
}

export function getSessionLogs(sessionId, { limit = 50, signal } = {}) {
  return requestJson(getApiBase(), `/sessions/${sessionId}/logs?limit=${encodeURIComponent(limit)}`, { signal })
}

export function createTask({ session_id, description, category, story_options, auto_start }, { signal } = {}) {
  return requestJson(getApiBase(), '/tasks', {
    method: 'POST',
    body: { session_id, description, category, story_options, auto_start },
    signal,
  })
}

export function getModelConfig({ signal } = {}) {
  return requestJson(getApiBase(), '/models/config', { signal })
}


export function updateModelConfig(config, { signal } = {}) {
  return requestJson(getApiBase(), '/models/config', {
    method: 'POST',
    body: config,
    signal,
  })
}

export function getFsTree(sessionId, { signal } = {}) {
  return requestJson(getApiBase(), `/fs/tree?session_id=${sessionId}`, { signal })
}

export function getFsContent(sessionId, path, { signal } = {}) {
  return requestJson(getApiBase(), `/fs/content?session_id=${sessionId}&path=${encodeURIComponent(path)}`, { signal })
}

export function saveFsContent(sessionId, path, content, { signal } = {}) {
  return requestJson(getApiBase(), '/fs/content', {
    method: 'POST',
    body: { session_id: sessionId, path, content },
    signal,
  })
}

export function refineChat(sessionId, path, message, { signal } = {}) {
  return requestJson(getApiBase(), '/chat/refine', {
    method: 'POST',
    body: { session_id: sessionId, path, message },
    signal,
  })
}

export function resumeTask(taskId, { signal } = {}) {
  return requestJson(getApiBase(), `/tasks/${taskId}/resume`, {
    method: 'POST',
    signal,
  })
}

export function submitTaskDecision(taskId, decision, proposal = '', { signal } = {}) {
  return requestJson(getApiBase(), `/tasks/${taskId}/decision`, {
    method: 'POST',
    body: { decision, proposal },
    signal,
  })
}

export function pauseTask(taskId, { force = false, signal } = {}) {
  const q = force ? '?force=true' : ''
  return requestJson(getApiBase(), `/tasks/${taskId}/pause${q}`, {
    method: 'POST',
    signal,
  })
}

export function restartTask(taskId, { signal } = {}) {
  return requestJson(getApiBase(), `/tasks/${taskId}/restart`, {
    method: 'POST',
    signal,
  })
}

export function deleteTask(taskId, { force = false, signal } = {}) {
  const q = force ? '?force=true' : ''
  return requestJson(getApiBase(), `/tasks/${taskId}${q}`, {
    method: 'DELETE',
    signal,
  })
}

export function refineTask(taskId, message, { autoStart = true, signal } = {}) {
  return requestJson(getApiBase(), `/tasks/${taskId}/refine`, {
    method: 'POST',
    body: { message, auto_start: autoStart },
    signal,
  })
}
