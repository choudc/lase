import { RefreshCw, ExternalLink, Eye, Loader2 } from 'lucide-react';
import { useState } from 'react';
import { createTask } from '@/api/lase';

export function Preview({ url = "about:blank", sessionId }) {
    const [key, setKey] = useState(0);
    const [inspecting, setInspecting] = useState(false);
    const isImageUrl = /\.(png|jpg|jpeg|webp)(\?.*)?$/i.test(url) || /\/api\/images\/[^/]+$/i.test(url);

    const reload = () => setKey(k => k + 1);

    const handleInspect = async () => {
        if (!sessionId || inspecting) return;
        setInspecting(true);
        try {
            await createTask({
                session_id: sessionId,
                description: `Take a screenshot of ${url} and visually inspect the UI. Critique the design against modern standards (spacing, typography, color harmony) and suggest improvements.`,
                auto_start: true
            });
            // Ideally show a toast here
        } catch (error) {
            console.error("Failed to start inspection task:", error);
        } finally {
            setTimeout(() => setInspecting(false), 2000); // Simple debounce/feedback
        }
    };

    return (
        <div className="h-full flex flex-col bg-white rounded-lg overflow-hidden border border-gray-200">
            <div className="h-10 bg-gray-100 border-b border-gray-200 flex items-center px-4 space-x-2">
                <div className="flex space-x-1.5 group">
                    <div className="w-3 h-3 rounded-full bg-red-400" />
                    <div className="w-3 h-3 rounded-full bg-yellow-400" />
                    <div className="w-3 h-3 rounded-full bg-green-400" />
                </div>
                <div className="flex-1 ml-4 bg-white h-7 rounded text-xs text-gray-500 flex items-center px-3 border border-gray-200 truncate">
                    {url}
                </div>

                {sessionId && (
                    <button
                        onClick={handleInspect}
                        disabled={inspecting}
                        className="p-1 hover:bg-indigo-100 text-indigo-600 rounded flex items-center space-x-1 transition-colors"
                        title="AI Visual Inspection"
                    >
                        {inspecting ? <Loader2 size={14} className="animate-spin" /> : <Eye size={14} />}
                        <span className="text-[10px] font-medium hidden md:inline">Inspect</span>
                    </button>
                )}

                <button onClick={reload} className="p-1 hover:bg-gray-200 rounded text-gray-600">
                    <RefreshCw size={14} />
                </button>
                <a href={url} target="_blank" rel="noopener noreferrer" className="p-1 hover:bg-gray-200 rounded text-gray-600">
                    <ExternalLink size={14} />
                </a>
            </div>
            <div className="flex-1 bg-white relative">
                {isImageUrl ? (
                    <img
                        key={key}
                        src={url}
                        alt="Generated preview"
                        className="w-full h-full object-contain bg-gray-950"
                    />
                ) : (
                    <iframe
                        key={key}
                        src={url}
                        className="w-full h-full border-0"
                        title="App Preview"
                    />
                )}
            </div>
        </div>
    );
}
