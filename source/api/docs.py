"""Swagger UI page (``/docs``) with direct file downloads.

FastAPI's stock page lets Swagger UI read every file response into a browser ``Blob``
before it shows "Download file". Chrome refuses large blobs when the drive holding its
profile is almost full (verified here: C: at 1.9 GB free of 230 GB), and Swagger then
reports ``Error: OK`` even though the server answered ``200`` with the file.

This page is FastAPI's default Swagger UI plus two small hooks:

* ``requestInterceptor`` installs a fetch wrapper: when a response is an attachment
  (``Content-Disposition: attachment``, e.g. ``/jobs/{job_id}/video``), the body is not
  buffered in the page; Swagger receives the real status and headers with a tiny body.
* a plugin renders "Download file" for attachment responses as a plain link to the
  request URL, so the browser's download manager saves the file (no Blob involved).

JSON responses (including 404/409 errors) are untouched.
"""
from __future__ import annotations

import html
import json

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter(include_in_schema=False)

# Same Swagger UI assets FastAPI's default /docs page loads.
_SWAGGER_JS = "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"
_SWAGGER_CSS = "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css"
_FAVICON = "https://fastapi.tiangolo.com/img/favicon.png"

_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link type="text/css" rel="stylesheet" href="__CSS__">
<link rel="shortcut icon" href="__FAVICON__">
<title>__TITLE__</title>
</head>
<body>
<div id="swagger-ui"></div>
<script src="__JS__"></script>
<script>
const isAttachment = (headers) => {
  const cd = headers && (headers.get ? headers.get("content-disposition")
                                     : headers["content-disposition"] || headers["Content-Disposition"]);
  return !!cd && /attachment/i.test(cd);
};

// Fetch used for API calls: attachments are not buffered in the page (see module docstring).
async function attachmentAwareFetch(url, options) {
  const response = await fetch(url, options);
  if (!response.ok || !isAttachment(response.headers)) return response;
  try { await response.body?.cancel(); } catch (e) { /* already closed */ }
  return new Response(" ", {status: response.status, statusText: response.statusText,
                            headers: response.headers});
}

// "Download file" for attachments links to the real URL (browser download manager).
const DirectDownloadPlugin = (system) => ({
  wrapComponents: {
    responseBody: (Original) => (props) => {
      const React = system.React;
      if (props.url && isAttachment(props.headers)) {
        const cd = props.headers["content-disposition"] || props.headers["Content-Disposition"];
        const match = /filename="?([^";]+)"?/i.exec(cd);
        return React.createElement("div", null,
          React.createElement("a", {href: props.url, download: match ? match[1] : ""},
                              "Download file"));
      }
      return React.createElement(Original, props);
    },
  },
});

window.ui = SwaggerUIBundle({
  url: __OPENAPI_URL__,
  dom_id: "#swagger-ui",
  layout: "BaseLayout",
  deepLinking: true,
  showExtensions: true,
  showCommonExtensions: true,
  presets: [SwaggerUIBundle.presets.apis, SwaggerUIBundle.SwaggerUIStandalonePreset],
  plugins: [DirectDownloadPlugin],
  requestInterceptor: (request) => {
    if (!request.loadSpec) request.userFetch = attachmentAwareFetch;
    return request;
  },
});
</script>
</body>
</html>
"""


@router.get("/docs", response_class=HTMLResponse)
def swagger_ui(request: Request) -> HTMLResponse:
    app = request.app
    title = html.escape(f"{app.title} - Swagger UI")
    page = (_PAGE.replace("__CSS__", _SWAGGER_CSS).replace("__JS__", _SWAGGER_JS)
            .replace("__FAVICON__", _FAVICON).replace("__TITLE__", title)
            .replace("__OPENAPI_URL__", json.dumps(app.openapi_url)))
    return HTMLResponse(page)
