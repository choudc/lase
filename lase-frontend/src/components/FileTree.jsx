import { useState } from 'react'
import {
    ChevronRight,
    ChevronDown,
    Folder,
    File,
    FileCode,
    FileJson,
    FileText
} from 'lucide-react'

const getFileIcon = (filename) => {
    if (filename.endsWith('.js') || filename.endsWith('.jsx') || filename.endsWith('.ts') || filename.endsWith('.tsx')) return <FileCode className="h-4 w-4 text-blue-500" />
    if (filename.endsWith('.json')) return <FileJson className="h-4 w-4 text-yellow-500" />
    if (filename.endsWith('.md') || filename.endsWith('.txt')) return <FileText className="h-4 w-4 text-gray-400" />
    return <File className="h-4 w-4 text-gray-400" />
}

const FileTreeNode = ({ node, onSelect, selectedPath, level = 0 }) => {
    const [isOpen, setIsOpen] = useState(false)

    const handleToggle = (e) => {
        e.stopPropagation()
        setIsOpen(!isOpen)
    }

    const handleSelect = (e) => {
        e.stopPropagation()
        if (node.type === 'file') {
            onSelect(node)
        } else {
            setIsOpen(!isOpen)
        }
    }

    return (
        <div>
            <div
                className={`flex items-center py-1 px-2 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800 ${selectedPath === node.id ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400' : ''
                    }`}
                style={{ paddingLeft: `${level * 12 + 8}px` }}
                onClick={handleSelect}
            >
                <span className="mr-1 flex-none" onClick={handleToggle}>
                    {node.type === 'dir' ? (
                        isOpen ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />
                    ) : <span className="w-3" />}
                </span>

                <span className="mr-2 flex-none">
                    {node.type === 'dir' ? (
                        <Folder className={`h-4 w-4 ${isOpen ? 'text-blue-500' : 'text-blue-400'}`} />
                    ) : getFileIcon(node.text)}
                </span>

                <span className="text-sm truncate select-none">{node.text}</span>
            </div>

            {isOpen && node.children && (
                <div>
                    {node.children.map(child => (
                        <FileTreeNode
                            key={child.id}
                            node={child}
                            onSelect={onSelect}
                            selectedPath={selectedPath}
                            level={level + 1}
                        />
                    ))}
                </div>
            )}
        </div>
    )
}

export function FileTree({ data, onSelectFile, selectedFile }) {
    if (!data || data.length === 0) {
        return <div className="p-4 text-sm text-gray-500 text-center">No files found</div>
    }

    return (
        <div className="py-2">
            {data.map(node => (
                <FileTreeNode
                    key={node.id}
                    node={node}
                    onSelect={onSelectFile}
                    selectedPath={selectedFile?.id}
                />
            ))}
        </div>
    )
}
