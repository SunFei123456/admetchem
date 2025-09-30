<template>
  <Teleport to="body">
    <Transition
      name="message"
      enter-active-class="transition-all duration-300 ease-out"
      enter-from-class="opacity-0 translate-y-2 scale-95"
      enter-to-class="opacity-100 translate-y-0 scale-100"
      leave-active-class="transition-all duration-200 ease-in"
      leave-from-class="opacity-100 translate-y-0 scale-100"
      leave-to-class="opacity-0 -translate-y-2 scale-95"
    >
      <div
        v-if="visible"
        :class="[
          'fixed top-4 right-4 z-50 max-w-sm w-full bg-white rounded-lg shadow-lg border-l-4 p-4',
          typeClasses
        ]"
      >
        <div class="flex items-start">
          <div class="flex-shrink-0">
            <i :class="iconClasses" class="text-lg"></i>
          </div>
          <div class="ml-3 flex-1">
            <h3 v-if="title" class="text-sm font-medium text-gray-900 mb-1">
              {{ title }}
            </h3>
            <p class="text-sm text-gray-700">
              {{ message }}
            </p>
          </div>
          <div class="ml-4 flex-shrink-0">
            <button
              @click="close"
              class="inline-flex text-gray-400 hover:text-gray-600 focus:outline-none focus:text-gray-600 transition ease-in-out duration-150"
            >
              <i class="fas fa-times text-sm"></i>
            </button>
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'

const props = defineProps({
  type: {
    type: String,
    default: 'info',
    validator: (value) => ['success', 'error', 'warning', 'info'].includes(value)
  },
  title: {
    type: String,
    default: ''
  },
  message: {
    type: String,
    required: true
  },
  duration: {
    type: Number,
    default: 3000
  },
  closable: {
    type: Boolean,
    default: true
  }
})

const emit = defineEmits(['close'])

const visible = ref(false)
let timer = null

const typeClasses = computed(() => {
  const classes = {
    success: 'border-green-400 bg-green-50',
    error: 'border-red-400 bg-red-50',
    warning: 'border-yellow-400 bg-yellow-50',
    info: 'border-blue-400 bg-blue-50'
  }
  return classes[props.type]
})

const iconClasses = computed(() => {
  const classes = {
    success: 'fas fa-check-circle text-green-500',
    error: 'fas fa-exclamation-circle text-red-500',
    warning: 'fas fa-exclamation-triangle text-yellow-500',
    info: 'fas fa-info-circle text-blue-500'
  }
  return classes[props.type]
})

const show = () => {
  visible.value = true
  if (props.duration > 0) {
    timer = setTimeout(() => {
      close()
    }, props.duration)
  }
}

const close = () => {
  visible.value = false
  if (timer) {
    clearTimeout(timer)
    timer = null
  }
  emit('close')
}

onMounted(() => {
  show()
})

onUnmounted(() => {
  if (timer) {
    clearTimeout(timer)
  }
})

defineExpose({
  show,
  close
})
</script>

<style scoped>
/* 动画样式已在 Transition 组件中定义 */
</style>