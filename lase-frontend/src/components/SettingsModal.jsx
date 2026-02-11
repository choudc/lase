import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogTitle,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
    Select,
    SelectContent,
    SelectGroup,
    SelectItem,
    SelectLabel,
    SelectTrigger,
    SelectValue,
} from '@/components/ui/select';
import { getModelConfig, updateModelConfig, getModels } from '@/api/lase';
import { AlertCircle, CheckCircle, Loader2, RefreshCw } from 'lucide-react';

export function SettingsModal({ open, onOpenChange }) {
    const [activeTab, setActiveTab] = useState('models');
    const [config, setConfig] = useState(null);
    const [availableModels, setAvailableModels] = useState([]);
    const [loading, setLoading] = useState(false);
    const [saving, setSaving] = useState(false);
    const [error, setError] = useState(null);
    const [success, setSuccess] = useState(false);

    useEffect(() => {
        if (open) {
            loadConfig();
        }
    }, [open]);

    const loadConfig = async () => {
        setLoading(true);
        setError(null);
        try {
            const [configData, modelsData] = await Promise.all([
                getModelConfig(),
                getModels()
            ]);
            setConfig(configData);
            setAvailableModels(modelsData);
        } catch (err) {
            const msg = err.body && err.body.error ? err.body.error : (err.message || String(err));
            setError(`Failed to load configuration: ${msg}`);
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const handleSave = async () => {
        setSaving(true);
        setError(null);
        setSuccess(false);
        try {
            await updateModelConfig(config);
            setSuccess(true);
            setTimeout(() => setSuccess(false), 2000);
        } catch (err) {
            const msg = err?.body?.error || err?.message || 'Failed to save configuration.';
            setError(msg);
            console.error(err);
        } finally {
            setSaving(false);
        }
    };

    const updateNestedConfig = (section, key, value) => {
        setConfig(prev => ({
            ...prev,
            [section]: {
                ...prev[section],
                [key]: value
            }
        }));
    };

    const updateDefaultModel = (taskType, field, value) => {
        setConfig(prev => ({
            ...prev,
            default_models: {
                ...prev.default_models,
                [taskType]: {
                    ...prev.default_models[taskType],
                    [field]: value
                }
            }
        }));
    }

    const handleModelSelect = (taskType, value) => {
        if (!value) return;
        const [provider, modelName] = value.split(':');
        setConfig(prev => ({
            ...prev,
            default_models: {
                ...prev.default_models,
                [taskType]: {
                    provider: provider,
                    name: modelName
                }
            }
        }));
    };

    // Group models by provider/group
    const groupedModels = availableModels.reduce((acc, model) => {
        const group = model.group || model.provider;
        if (!acc[group]) acc[group] = [];
        acc[group].push(model);
        return acc;
    }, {});

    if (!config && loading) return null;

    return (
        <Dialog open={open} onOpenChange={onOpenChange}>
            <DialogContent className="sm:max-w-[700px] h-[80vh] flex flex-col">
                <DialogHeader>
                    <DialogTitle>Settings</DialogTitle>
                    <DialogDescription>
                        Configure AI models, API keys, and agent preferences.
                    </DialogDescription>
                </DialogHeader>

                {config && (
                    <Tabs defaultValue="models" className="flex-1 overflow-hidden flex flex-col" value={activeTab} onValueChange={setActiveTab}>
                        <TabsList className="grid w-full grid-cols-2">
                            <TabsTrigger value="models">Model Selection</TabsTrigger>
                            <TabsTrigger value="keys">API Keys</TabsTrigger>
                        </TabsList>

                        <TabsContent value="models" className="flex-1 overflow-y-auto p-4 space-y-4">
                            <div className="space-y-4">
                                <div className="flex items-center justify-between">
                                    <h3 className="text-sm font-medium text-gray-500">Task Specific Models</h3>
                                    <Button variant="ghost" size="sm" onClick={loadConfig} disabled={loading}>
                                        <RefreshCw className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
                                        Refresh Models
                                    </Button>
                                </div>

                                {['coding', 'general', 'vision', 'reasoning'].map(task => {
                                    const currentProvider = config.default_models?.[task]?.provider;
                                    const currentName = config.default_models?.[task]?.name;
                                    const currentValue = currentProvider && currentName ? `${currentProvider}:${currentName}` : '';

                                    // Check if current value exists in available models
                                    const exists = availableModels.some(m => `${m.provider}:${m.name}` === currentValue);

                                    // If not exists and we have a value, add a "Configured" group
                                    let displayGroups = { ...groupedModels };
                                    if (currentValue && !exists) {
                                        if (!displayGroups['Configured']) displayGroups['Configured'] = [];
                                        const alreadyAdded = displayGroups['Configured'].some(m => m.name === currentName && m.provider === currentProvider);
                                        if (!alreadyAdded) {
                                            displayGroups['Configured'].push({
                                                provider: currentProvider,
                                                name: currentName,
                                                group: 'Configured'
                                            });
                                        }
                                    }

                                    return (
                                        <div key={task} className="grid grid-cols-1 md:grid-cols-4 gap-4 items-center border p-3 rounded-lg">
                                            <Label className="capitalize col-span-1">{task} Model</Label>
                                            <div className="col-span-3">
                                                <Select
                                                    value={currentValue}
                                                    onValueChange={(val) => handleModelSelect(task, val)}
                                                >
                                                    <SelectTrigger>
                                                        <SelectValue placeholder="Select a model" />
                                                    </SelectTrigger>
                                                    <SelectContent>
                                                        {Object.entries(displayGroups).map(([group, models]) => (
                                                            <SelectGroup key={group}>
                                                                <SelectLabel>{group}</SelectLabel>
                                                                {models.map(model => (
                                                                    <SelectItem
                                                                        key={`${model.provider}:${model.name}`}
                                                                        value={`${model.provider}:${model.name}`}
                                                                    >
                                                                        {model.name}
                                                                    </SelectItem>
                                                                ))}
                                                            </SelectGroup>
                                                        ))}
                                                    </SelectContent>
                                                </Select>
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        </TabsContent>

                        <TabsContent value="keys" className="flex-1 overflow-y-auto p-4 space-y-4">
                            <div className="space-y-4">
                                <div className="space-y-2">
                                    <Label>OpenAI API Key</Label>
                                    <Input
                                        type="password"
                                        value={config.openai_settings?.api_key || ''}
                                        onChange={(e) => updateNestedConfig('openai_settings', 'api_key', e.target.value)}
                                        placeholder="sk-..."
                                    />
                                </div>
                                <div className="space-y-2">
                                    <Label>Gemini API Key</Label>
                                    <Input
                                        type="password"
                                        value={config.gemini_settings?.api_key || ''}
                                        onChange={(e) => updateNestedConfig('gemini_settings', 'api_key', e.target.value)}
                                        placeholder="AIza..."
                                    />
                                </div>
                                <div className="space-y-2">
                                    <Label>Anthropic API Key</Label>
                                    <Input
                                        type="password"
                                        value={config.anthropic_settings?.api_key || ''}
                                        onChange={(e) => updateNestedConfig('anthropic_settings', 'api_key', e.target.value)}
                                        placeholder="sk-ant..."
                                    />
                                </div>
                                <div className="space-y-2">
                                    <Label>DeepSeek API Key</Label>
                                    <Input
                                        type="password"
                                        value={config.deepseek_settings?.api_key || ''}
                                        onChange={(e) => updateNestedConfig('deepseek_settings', 'api_key', e.target.value)}
                                        placeholder="sk-..."
                                    />
                                </div>
                                <div className="space-y-2">
                                    <Label>Stability API Key</Label>
                                    <Input
                                        type="password"
                                        value={config.stability_settings?.api_key || ''}
                                        onChange={(e) => updateNestedConfig('stability_settings', 'api_key', e.target.value)}
                                        placeholder="sk-..."
                                    />
                                </div>
                                <div className="space-y-2 pt-4 border-t">
                                    <Label>Ollama Base URL</Label>
                                    <Input
                                        value={config.ollama_settings?.base_url || 'http://localhost:11434'}
                                        onChange={(e) => updateNestedConfig('ollama_settings', 'base_url', e.target.value)}
                                    />
                                </div>
                                <div className="space-y-2">
                                    <Label>Stability Base URL</Label>
                                    <Input
                                        value={config.stability_settings?.base_url || 'https://api.stability.ai'}
                                        onChange={(e) => updateNestedConfig('stability_settings', 'base_url', e.target.value)}
                                    />
                                </div>
                                <div className="space-y-2">
                                    <Label>Stability Default Aspect Ratio</Label>
                                    <Input
                                        value={config.stability_settings?.default_aspect_ratio || '1:1'}
                                        onChange={(e) => updateNestedConfig('stability_settings', 'default_aspect_ratio', e.target.value)}
                                        placeholder="e.g. 1:1, 16:9, 9:16"
                                    />
                                </div>
                                <div className="space-y-2">
                                    <Label>Stability Default Style Preset</Label>
                                    <Input
                                        value={config.stability_settings?.default_style_preset || ''}
                                        onChange={(e) => updateNestedConfig('stability_settings', 'default_style_preset', e.target.value)}
                                        placeholder="e.g. photographic, anime, cinematic"
                                    />
                                </div>
                                <div className="space-y-2">
                                    <Label>Stability Output Format</Label>
                                    <Input
                                        value={config.stability_settings?.default_output_format || 'png'}
                                        onChange={(e) => updateNestedConfig('stability_settings', 'default_output_format', e.target.value)}
                                        placeholder="png | jpeg | webp"
                                    />
                                </div>
                            </div>
                        </TabsContent>
                    </Tabs>
                )}

                <DialogFooter className="mt-4">
                    <div className="flex-1 flex items-center text-sm">
                        {error && <span className="text-red-500 flex items-center"><AlertCircle className="w-4 h-4 mr-1" /> {error}</span>}
                        {success && <span className="text-green-500 flex items-center"><CheckCircle className="w-4 h-4 mr-1" /> Saved!</span>}
                    </div>
                    <Button onClick={handleSave} disabled={saving}>
                        {saving && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                        Save Changes
                    </Button>
                </DialogFooter>
            </DialogContent>
        </Dialog>
    );
}
