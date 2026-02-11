import { useState } from 'react';
import { Check, Copy } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';

export function TaskOutput({ content }) {
    if (!content) return null;

    // Simple parser for markdown code blocks
    const parseContent = (text) => {
        const parts = [];
        // Regex to find ```language ... ``` blocks
        // Captures: 1=language (optional), 2=content
        const regex = /```(\w*)\n([\s\S]*?)```/g;

        let lastIndex = 0;
        let match;

        while ((match = regex.exec(text)) !== null) {
            // Add preceding text
            if (match.index > lastIndex) {
                parts.push({
                    type: 'text',
                    content: text.slice(lastIndex, match.index)
                });
            }

            // Add code block
            parts.push({
                type: 'code',
                language: match[1] || 'text',
                content: match[2]
            });

            lastIndex = regex.lastIndex;
        }

        // Add remaining text
        if (lastIndex < text.length) {
            parts.push({
                type: 'text',
                content: text.slice(lastIndex)
            });
        }

        return parts;
    };

    const segments = parseContent(content);

    return (
        <div className="space-y-4 text-sm">
            {segments.map((segment, index) => {
                if (segment.type === 'code') {
                    return <CodeBlock key={index} language={segment.language} code={segment.content} />;
                }
                return (
                    <div key={index} className="whitespace-pre-wrap text-gray-700 dark:text-gray-300 leading-relaxed">
                        {segment.content}
                    </div>
                );
            })}
        </div>
    );
}

function CodeBlock({ language, code }) {
    const [copied, setCopied] = useState(false);

    const onCopy = () => {
        navigator.clipboard.writeText(code);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    return (
        <div className="rounded-md border bg-gray-950 dark:bg-gray-950 overflow-hidden my-2">
            <div className="flex items-center justify-between px-4 py-2 bg-gray-900 border-b border-gray-800">
                <span className="text-xs font-medium text-gray-400 lowercase">{language}</span>
                <Button variant="ghost" size="icon" className="h-6 w-6 text-gray-400 hover:text-white" onClick={onCopy}>
                    {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
                </Button>
            </div>
            <ScrollArea className="w-full">
                <div className="p-4 overflow-x-auto">
                    <pre className="font-mono text-xs text-gray-300 leading-normal">
                        {code}
                    </pre>
                </div>
            </ScrollArea>
        </div>
    );
}
