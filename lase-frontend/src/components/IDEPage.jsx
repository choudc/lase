import { useState, useEffect, useRef } from 'react'
import {
    Folder,
    Save,
    MessageSquare,
    Play,
    RotateCcw,
    Sparkles,
    RefreshCw,
    X
} from 'lucide-react'
import { Button } from '@/components/ui/button.jsx'
import { Textarea } from '@/components/ui/textarea.jsx'
import { ScrollArea } from '@/components/ui/scroll-area.jsx'
import {
    Panel,
    PanelGroup,
    PanelResizeHandle,
} from "react-resizable-panels"
import { Badge } from '@/components/ui/badge.jsx'

import {
    getFsTree,
    getFsContent,
    saveFsContent,
    refineChat
} from '@/api/lase'
import { FileTree } from './FileTree'

export function IDEPage({ session, onClose }) {
    const [fileTree, setFileTree] = useState([])
    const [selectedFile, setSelectedFile] = useState(null)
    const [fileContent, setFileContent] = useState('')
    const [isDirty, setIsDirty] = useState(false)
    const [isLoadingTree, setIsLoadingTree] = useState(false)
    const [isLoadingContent, setIsLoadingContent] = useState(false)
    const [isSaving, setIsSaving] = useState(false)

    // Chat state
    const [chatMessage, setChatMessage] = useState('')
    const [isRefining, setIsRefining] = useState(false)
    const [chatHistory, setChatHistory] = useState([])

    const loadTree = async () => {
        setIsLoadingTree(true)
        try {
            const tree = await getFsTree(session.id)
            setFileTree(tree)
        } catch (error) {
            console.error("Failed to load file tree:", error)
        } finally {
            setIsLoadingTree(false)
        }
    }

    useEffect(() => {
        if (session?.id) {
            loadTree()
        }
    }, [session?.id])

    const handleSelectFile = async (node) => {
        if (selectedFile?.id === node.id) return
        if (isDirty) {
            if (!confirm("You have unsaved changes. Discard them?")) return
        }

        setSelectedFile(node)
        setIsLoadingContent(true)
        setIsDirty(false)
        try {
            const data = await getFsContent(session.id, node.id)
            setFileContent(data.content || "")
        } catch (error) {
            console.error("Failed to load content:", error)
            setFileContent("// Error loading file content")
        } finally {
            setIsLoadingContent(false)
        }
    }

    const handleSave = async () => {
        if (!selectedFile) return
        setIsSaving(true)
        try {
            await saveFsContent(session.id, selectedFile.id, fileContent)
            setIsDirty(false)
        } catch (error) {
            alert("Failed to save: " + error.message)
        } finally {
            setIsSaving(false)
        }
    }

    const handleRefine = async () => {
        if (!chatMessage.trim()) return
        setIsRefining(true)

        // Optimistic UI
        const newUserMsg = { role: 'user', content: chatMessage }
        setChatHistory(prev => [...prev, newUserMsg])
        setChatMessage('')

        try {
            const task = await refineChat(session.id, selectedFile?.id, chatMessage)

            const newAgentMsg = {
                role: 'agent',
                content: `I've started a task to help you: "${task.description}".\nTask ID: ${task.id.slice(0, 8)}`
            }
            setChatHistory(prev => [...prev, newAgentMsg])
        } catch (error) {
            setChatHistory(prev => [...prev, { role: 'error', content: "Failed to start refinement: " + error.message }])
        } finally {
            setIsRefining(false)
        }
    }

    return (
        <div className="fixed inset-0 z-50 bg-white dark:bg-gray-900 flex flex-col">
            {/* Toolbar */}
            <div className="h-12 border-b dark:border-gray-800 flex items-center justify-between px-4 bg-gray-50 dark:bg-gray-900">
                <div className="flex items-center space-x-4">
                    <Button variant="ghost" size="sm" onClick={onClose} className="mr-2">
                        <X className="h-4 w-4 mr-2" /> Exit IDE
                    </Button>
                    <span className="font-semibold text-sm">{session.name}</span>
                    <span className="text-xs text-gray-500 font-mono hidden md:inline-block">
                        {selectedFile ? selectedFile.text : "No file selected"}
                    </span>
                    {isDirty && <Badge variant="secondary" className="bg-yellow-100 text-yellow-800">Unsaved</Badge>}
                </div>

                <div className="flex items-center space-x-2">
                    <Button variant="outline" size="sm" onClick={loadTree} title="Refresh File Tree">
                        <RefreshCw className={`h-4 w-4 ${isLoadingTree ? 'animate-spin' : ''}`} />
                    </Button>
                    <Button
                        size="sm"
                        onClick={handleSave}
                        disabled={!selectedFile || !isDirty || isSaving}
                        className="bg-blue-600 hover:bg-blue-700"
                    >
                        <Save className="h-4 w-4 mr-2" />
                        {isSaving ? "Saving..." : "Save File"}
                    </Button>
                </div>
            </div>

            {/* Main Content */}
            <div className="flex-1 overflow-hidden">
                <PanelGroup direction="horizontal">

                    {/* File Explorer */}
                    <Panel defaultSize={20} minSize={15} maxSize={30} className="border-r dark:border-gray-800 bg-gray-50 dark:bg-gray-900/50">
                        <div className="flex flex-col h-full">
                            <div className="px-4 py-2 text-xs font-semibold text-gray-500 uppercase tracking-wider">Explorer</div>
                            <ScrollArea className="flex-1 px-2">
                                <FileTree
                                    data={fileTree}
                                    onSelectFile={handleSelectFile}
                                    selectedFile={selectedFile}
                                />
                            </ScrollArea>
                        </div>
                    </Panel>

                    <PanelResizeHandle className="w-1 bg-gray-200 dark:bg-gray-800 hover:bg-blue-500 transition-colors" />

                    {/* Editor Area */}
                    <Panel defaultSize={50} minSize={30}>
                        <div className="h-full flex flex-col relative">
                            {isLoadingContent && (
                                <div className="absolute inset-0 bg-white/50 dark:bg-black/50 z-10 flex items-center justify-center">
                                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                                </div>
                            )}

                            {!selectedFile ? (
                                <div className="flex-1 flex items-center justify-center text-gray-400 flex-col">
                                    <Folder className="h-16 w-16 mb-4 opacity-20" />
                                    <p>Select a file to view or edit</p>
                                </div>
                            ) : (
                                <textarea
                                    className="flex-1 w-full h-full p-4 font-mono text-sm resize-none focus:outline-none bg-white dark:bg-gray-950 text-gray-800 dark:text-gray-200 leading-relaxed"
                                    value={fileContent}
                                    onChange={(e) => {
                                        setFileContent(e.target.value)
                                        setIsDirty(true)
                                    }}
                                    spellCheck={false}
                                />
                            )}
                        </div>
                    </Panel>

                    <PanelResizeHandle className="w-1 bg-gray-200 dark:bg-gray-800 hover:bg-blue-500 transition-colors" />

                    {/* Chat / Refinement Panel */}
                    <Panel defaultSize={30} minSize={20} className="border-l dark:border-gray-800 bg-white dark:bg-gray-900">
                        <div className="flex flex-col h-full">
                            <div className="px-4 py-2 border-b dark:border-gray-800 flex items-center justify-between">
                                <span className="text-xs font-semibold text-gray-500 uppercase tracking-wider flex items-center">
                                    <Sparkles className="h-3 w-3 mr-1 text-purple-500" />
                                    AI Architect
                                </span>
                            </div>

                            <ScrollArea className="flex-1 p-4">
                                <div className="space-y-4">
                                    {chatHistory.length === 0 && (
                                        <div className="text-center text-gray-400 text-sm mt-10">
                                            <p>Ask me to refine this file.</p>
                                            <p className="text-xs mt-2">"Add error handling"</p>
                                            <p className="text-xs">"Refactor this function"</p>
                                        </div>
                                    )}
                                    {chatHistory.map((msg, i) => (
                                        <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                                            <div className={`max-w-[85%] rounded-lg p-3 text-sm ${msg.role === 'user'
                                                    ? 'bg-blue-600 text-white'
                                                    : msg.role === 'error'
                                                        ? 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300'
                                                        : 'bg-gray-100 dark:bg-gray-800 text-gray-800 dark:text-gray-200'
                                                }`}>
                                                {msg.content}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </ScrollArea>

                            <div className="p-4 border-t dark:border-gray-800">
                                <div className="flex gap-2">
                                    <Textarea
                                        placeholder={selectedFile ? `Ask to change ${selectedFile.text}...` : "Select a file to refine..."}
                                        value={chatMessage}
                                        onChange={e => setChatMessage(e.target.value)}
                                        className="min-h-[60px] resize-none"
                                        disabled={!selectedFile}
                                    />
                                    <Button
                                        size="icon"
                                        className="h-[60px] w-[60px]"
                                        onClick={handleRefine}
                                        disabled={!selectedFile || !chatMessage.trim() || isRefining}
                                    >
                                        {isRefining ? <RotateCcw className="animate-spin" /> : <Play className="ml-1" />}
                                    </Button>
                                </div>
                                {!selectedFile && <p className="text-xs text-orange-500 mt-2">Select a file to start a refinement chat.</p>}
                            </div>
                        </div>
                    </Panel>

                </PanelGroup>
            </div>
        </div>
    )
}
