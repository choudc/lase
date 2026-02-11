function joinUrl(base, path) {
  const b = String(base || '').replace(/\/+$/, '')
  const p = String(path || '').replace(/^\/+/, '')
  if (!b) return `/${p}`
  return `${b}/${p}`
}

export class HttpError extends Error {
  constructor(message, { status, statusText, body } = {}) {
    super(message)
    this.name = 'HttpError'
    this.status = status
    this.statusText = statusText
    this.body = body
  }
}

export async function requestJson(baseUrl, path, { method = 'GET', headers, body, signal } = {}) {
  const url = joinUrl(baseUrl, path)
  const res = await fetch(url, {
    method,
    headers: {
      Accept: 'application/json',
      ...(body ? { 'Content-Type': 'application/json' } : null),
      ...(headers || null),
    },
    body: body ? JSON.stringify(body) : undefined,
    signal,
  })

  const contentType = res.headers.get('content-type') || ''
  const isJson = contentType.includes('application/json')
  const payload = isJson ? await res.json().catch(() => null) : await res.text().catch(() => null)

  if (!res.ok) {
    throw new HttpError(`HTTP ${res.status} ${res.statusText}`, {
      status: res.status,
      statusText: res.statusText,
      body: payload,
    })
  }

  return payload
}

