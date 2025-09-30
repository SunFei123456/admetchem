<template>
  <div class="molecular-editor-container">
    <!-- 组件标题 -->
    <div class="mb-4">
      <h3 class="text-lg font-semibold text-gray-800 mb-2">Draw Molecule from Editor Below</h3>
      <p class="text-sm text-gray-600">Use the molecular editor to draw or modify chemical structures</p>
    </div>

    <!-- JSME 分子编辑器容器 -->
    <div class=" ">
      <div 
        id="jsme_container" 
        ref="jsmeContainer"
        class="molecular-editor-wrapper"
      >
   
      </div>
      
      <!-- 操作按钮 -->
      <div class="mt-4 flex justify-between items-center">
        <div class="flex space-x-2">
          <button 
            @click="clearEditor"
            class="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600 transition-colors duration-200 text-sm"
          >
            <i class="fas fa-trash mr-2"></i>Clear
          </button>
        </div>
        <div class="text-sm text-gray-600">
          <span>Ready to draw molecular structures</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount, defineEmits } from 'vue'

// Props
const props = defineProps({
  // 宽度支持数字或字符串（如 '100%'），默认填充父容器宽度
  width: { type: [String, Number], default: '100%' },
  height: { type: Number, default: 340 }
})

// Emits
const emit = defineEmits(['smiles-generated', 'structure-changed'])

// Refs
const jsmeContainer = ref(null)
const jsmeInstance = ref(null)

// 计算样式（保留现有样式绑定）
const editorStyle = {
  padding: '0px',
  outline: '0px',
  textAlign: 'left',
  width: typeof props.width === 'number' ? `${props.width}px` : String(props.width),
  height: `${props.height}px`
}

// 动态加载 JSME 脚本
function loadJsmeScript() {
  return new Promise((resolve, reject) => {
    // 若已加载则直接返回
    if (window.JSApplet && window.JSApplet.JSME) {
      return resolve()
    }

    // 定义官方要求的全局回调
    window.jsmeOnLoad = function () {
      resolve()
    }

    // 从官方 CDN 加载 JSME
    const src = 'https://jsme-editor.github.io/dist/jsme/jsme.nocache.js'
    // 防止重复插入脚本
    const exists = Array.from(document.getElementsByTagName('script')).some(s => s.src === src)
    const script = exists ? null : document.createElement('script')
    script.type = 'text/javascript'
    script.src = src
    // 当浏览器未触发 jsmeOnLoad 时兜底
    script && (script.onload = () => {
      // 如果浏览器未触发 jsmeOnLoad，这里兜底
      if (window.JSApplet && window.JSApplet.JSME) {
        resolve()
      }
    })
    script && (script.onerror = () => reject(new Error('Failed to load JSME script')))
    if (!exists) document.head.appendChild(script)
  })
}

// 初始化编辑器
async function initJsme() {
  await loadJsmeScript()

  const containerId = 'jsme_container'
  // 实例化 JSME（第三个参数为可选配置对象，这里使用默认）
  // 参考官方文档：https://jsme-editor.github.io/dist/doc.html#javascript_install
  // JSApplet.JSME(containerId, width, height, options?)
  const w = typeof props.width === 'number' ? `${props.width}px` : String(props.width)
  const h = `${props.height}px`
  // 确保容器存在
  if (!document.getElementById(containerId)) return

  // 创建实例
  // eslint-disable-next-line new-cap
  const instance = new window.JSApplet.JSME(containerId, w, h, { options: 'stereo' })
  jsmeInstance.value = instance

  // 注册结构变化回调：AfterStructureModified
  instance.setCallBack('AfterStructureModified', (jsmeEvent) => {
    try {
      const smiles = instance.smiles()
      emit('structure-changed', jsmeEvent)
      // 同步向外发射 SMILES，便于上层接入
      emit('smiles-generated', smiles)
    } catch (e) {
      // 忽略单次回调错误
      console.warn('JSME callback error:', e)
    }
  })
}

// 清空编辑器
function clearEditor() {
  const inst = jsmeInstance.value
  if (!inst) return
  // 尝试使用 reset（若不可用则回退为空分子）
  if (typeof inst.reset === 'function') {
    inst.reset()
  } else if (typeof inst.readMolFile === 'function') {
    inst.readMolFile('')
  }
  emit('structure-changed', '')
}


onMounted(() => {
  initJsme()
})

onBeforeUnmount(() => {
  // 可选：移除全局回调，防止重复触发
  if (window.jsmeOnLoad) delete window.jsmeOnLoad
})
</script>

<style scoped>
.molecular-editor-container {
  max-width: 100%;
}

.molecular-editor-wrapper {
  position: relative;
  overflow: hidden;
}

.jsme-editor {
  position: relative;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  background: #f9fafb;
}

.jsa-resetDiv {
  position: relative;
  overflow: hidden;
}

.editor-main {
  width: 100%;
  height: 100%;
}

.drawing-area {
  position: absolute;
  width: 356px;
  height: 267px;
  left: 24px;
  top: 49px;
  background: white;
  border: 1px solid #d1d5db;
}

.toolbar {
  position: absolute;
  width: 380px;
  height: 49px;
  left: 0px;
  top: 0px;
}

.element-panel {
  position: absolute;
  width: 24px;
  height: 293px;
  left: 0px;
  top: 49px;
}

.status-bar {
  position: absolute;
  width: 356px;
  height: 24px;
  left: 24px;
  top: 316px;
}

/* 悬停效果 */
button:hover {
  transform: translateY(-1px);
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

button:active {
  transform: translateY(0);
}

/* 响应式处理 */
@media (max-width: 640px) {
  .molecular-editor-wrapper {
    overflow-x: auto;
  }
  

}
</style>