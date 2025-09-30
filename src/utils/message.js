import { createApp } from 'vue'
import MessageBox from '@/components/MessageBox.vue'

// 消息队列
let messageQueue = []
let messageId = 0

// 创建消息实例
const createMessage = (options) => {
  const id = ++messageId
  
  // 创建容器
  const container = document.createElement('div')
  container.id = `message-${id}`
  document.body.appendChild(container)
  
  // 创建Vue应用实例
  const app = createApp(MessageBox, {
    ...options,
    onClose: () => {
      // 清理DOM
      setTimeout(() => {
        if (container && container.parentNode) {
          container.parentNode.removeChild(container)
        }
        app.unmount()
        
        // 从队列中移除
        const index = messageQueue.findIndex(item => item.id === id)
        if (index > -1) {
          messageQueue.splice(index, 1)
        }
      }, 300) // 等待动画完成
      
      if (options.onClose) {
        options.onClose()
      }
    }
  })
  
  // 挂载组件
  const instance = app.mount(container)
  
  // 添加到队列
  const messageItem = {
    id,
    instance,
    app,
    container,
    close: () => instance.close()
  }
  
  messageQueue.push(messageItem)
  
  return messageItem
}

// 主要的消息函数
const message = (options) => {
  if (typeof options === 'string') {
    options = { message: options }
  }
  
  return createMessage({
    type: 'info',
    duration: 3000,
    closable: true,
    ...options
  })
}

// 成功消息
message.success = (options) => {
  if (typeof options === 'string') {
    options = { message: options }
  }
  return createMessage({
    type: 'success',
    duration: 3000,
    closable: true,
    ...options
  })
}

// 错误消息
message.error = (options) => {
  if (typeof options === 'string') {
    options = { message: options }
  }
  return createMessage({
    type: 'error',
    duration: 4000,
    closable: true,
    ...options
  })
}

// 警告消息
message.warning = (options) => {
  if (typeof options === 'string') {
    options = { message: options }
  }
  return createMessage({
    type: 'warning',
    duration: 3000,
    closable: true,
    ...options
  })
}

// 信息消息
message.info = (options) => {
  if (typeof options === 'string') {
    options = { message: options }
  }
  return createMessage({
    type: 'info',
    duration: 3000,
    closable: true,
    ...options
  })
}

// 关闭所有消息
message.closeAll = () => {
  messageQueue.forEach(item => {
    item.close()
  })
}

// 获取当前消息数量
message.getCount = () => {
  return messageQueue.length
}

export default message
export { message }