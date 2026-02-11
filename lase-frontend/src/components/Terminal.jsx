import { useEffect, useRef } from 'react';
import { Terminal as Xterm } from 'xterm';
import { FitAddon } from 'xterm-addon-fit';
import 'xterm/css/xterm.css';
import { io } from 'socket.io-client';

const SOCKET_URL = "http://127.0.0.1:5000";

export function Terminal() {
    const terminalRef = useRef(null);
    const xtermRef = useRef(null);
    const socketRef = useRef(null);

    useEffect(() => {
        if (!terminalRef.current) return;

        // Initialize Xterm
        const term = new Xterm({
            cursorBlink: true,
            theme: {
                background: '#1e1e1e',
                foreground: '#ffffff',
            },
            fontFamily: 'monospace',
            fontSize: 14,
        });

        const fitAddon = new FitAddon();
        term.loadAddon(fitAddon);

        term.open(terminalRef.current);
        fitAddon.fit();

        xtermRef.current = term;

        // Connect to SocketIO
        const socket = io(SOCKET_URL, {
            transports: ['websocket', 'polling'], // explicit
        });

        socket.on('connect', () => {
            term.writeln('\x1b[32mConnected to LASE Backend...\x1b[0m');
        });

        socket.on('process_log', (data) => {
            // { pid: "...", lines: ["..."] }
            const { pid, lines } = data;
            const safePid = String(pid || 'proc');
            const prefix = `[${safePid}] `;
            const indent = ' '.repeat(prefix.length);

            const normalizeNewlines = (text) =>
                String(text ?? '')
                    .replace(/\r\n/g, '\n')
                    .replace(/\r/g, '\n');

            const maybePrettyJson = (text) => {
                const t = String(text || '').trim();
                if (!(t.startsWith('{') || t.startsWith('['))) return text;
                try {
                    return JSON.stringify(JSON.parse(t), null, 2);
                } catch {
                    return text;
                }
            };

            lines.forEach((rawLine) => {
                const pretty = maybePrettyJson(rawLine);
                const parts = normalizeNewlines(pretty).split('\n');
                parts.forEach((part, index) => {
                    const content = part.length ? part : '';
                    if (index === 0) {
                        term.writeln(`\x1b[34m${prefix}\x1b[0m${content}`);
                    } else {
                        term.writeln(`${indent}${content}`);
                    }
                });
            });
        });

        socket.on('disconnect', () => {
            term.writeln('\x1b[31mDisconnected from LASE Backend.\x1b[0m');
        });

        socketRef.current = socket;

        // Handle resize
        const handleResize = () => fitAddon.fit();
        window.addEventListener('resize', handleResize);

        return () => {
            socket.disconnect();
            term.dispose();
            window.removeEventListener('resize', handleResize);
        };
    }, []);

    return (
        <div className="h-full w-full bg-black p-2 rounded-lg overflow-hidden border border-gray-800">
            <div ref={terminalRef} className="h-full w-full" />
        </div>
    );
}
