/* Pangolin GPU Planner – service worker (offline support for the installed PWA).
 * Bump CACHE when you change any cached file so clients pick up the new version. */
const CACHE = "pangolin-gpu-planner-v2.0.0";
const ASSETS = [
  "./", "./index.html", "./css/styles.css", "./js/data.js", "./js/app.js", "./manifest.webmanifest",
  "./icons/favicon.svg", "./icons/icon-192.png", "./icons/icon-512.png", "./icons/maskable-512.png", "./icons/apple-touch-icon.png"
];
self.addEventListener("install", e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(ASSETS)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", e => {
  e.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))).then(() => self.clients.claim()));
});
/* Network-first for HTML and data (so updates show when online), cache-first for everything else. */
self.addEventListener("fetch", e => {
  const req = e.request;
  if (req.method !== "GET" || new URL(req.url).origin !== location.origin) return;
  const fresh = req.mode === "navigate" || /\/js\/data\.js$/.test(new URL(req.url).pathname);
  if (fresh) {
    e.respondWith(fetch(req).then(r => { const copy = r.clone(); caches.open(CACHE).then(c => c.put(req, copy)); return r; })
      .catch(() => caches.match(req).then(r => r || caches.match("./index.html"))));
  } else {
    e.respondWith(caches.match(req).then(r => r || fetch(req).then(res => { const copy = res.clone(); caches.open(CACHE).then(c => c.put(req, copy)); return res; })));
  }
});
