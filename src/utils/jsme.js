/*
  JSME loader
  - 动态加载官方 CDN 脚本：https://jsme-editor.github.io/dist/jsme/jsme.nocache.js
  - 重复加载保护：若已存在或已加载则不再重复注入
  - 与 MolecularEditor.vue 的 window.jsmeOnLoad 回调兼容
*/
(function () {
  var CDN_URL = 'https://jsme-editor.github.io/dist/jsme/jsme.nocache.js'
  var SCRIPT_ID = 'jsme-cdn-script'

  // 若已存在全局对象，直接触发回调并返回
  if (window.JSApplet && window.JSApplet.JSME) {
    try {
      // 异步触发，保持调用栈整洁
      setTimeout(function () {
        if (typeof window.jsmeOnLoad === 'function') {
          window.jsmeOnLoad()
        }
      }, 0)
    } catch (e) {
      console.warn('JSME loader: onLoad callback error (already loaded):', e)
    }
    return
  }

  // 若正在加载或已注入过脚本，则不重复注入
  var existing = document.getElementById(SCRIPT_ID)
  if (existing) {
    return
  }

  // 注入 CDN 脚本
  var script = document.createElement('script')
  script.id = SCRIPT_ID
  script.type = 'text/javascript'
  script.async = true
  script.src = CDN_URL

  // 兜底：CDN onload 后若未调用 jsmeOnLoad，但全局对象已可用，则主动触发
  script.onload = function () {
    // 等待 JSME 初始化内部逻辑，如果 1 秒后仍未调用 jsmeOnLoad，则兜底
    setTimeout(function () {
      if (window.JSApplet && window.JSApplet.JSME && typeof window.jsmeOnLoad === 'function') {
        try {
          window.jsmeOnLoad()
        } catch (e) {
          console.warn('JSME loader: fallback jsmeOnLoad call error:', e)
        }
      }
    }, 1000)
  }

  script.onerror = function () {
    console.error('JSME loader: failed to load CDN script:', CDN_URL)
  }

  document.head.appendChild(script)
})()
