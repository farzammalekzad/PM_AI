# Task 16B: Offline Vazir Font Integration

The approved supplied assets are:

- `app/web/static/fonts/Vazir.woff2` — regular, WOFF2
- `app/web/static/fonts/Vazir-Bold.woff2` — bold, WOFF2

No accompanying license text was present in the supplied font directory. These files
were used solely because this task explicitly identifies them as approved local assets.

The stylesheet defines local `@font-face` entries for weights 400 and 700 with
`font-display: swap`. Vazir is applied to the root, body, form controls, buttons,
dialogs, chat, source citations, document management, and monitoring through the shared
font stack. Technical `<pre>` output retains a cross-platform monospace stack.

The static allowlist exposes only the exact two font paths, both with `font/woff2` MIME
type. The page CSP explicitly specifies `font-src 'self'`; no CDN, Google Fonts, or
external URL is referenced. Unknown font paths and traversal attempts remain blocked.
