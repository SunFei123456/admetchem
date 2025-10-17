<template>
    <div class="flex flex-col h-screen bg-gray-50">
        <!-- 聊天区域 -->
        <div ref="chatContainer" class="flex-1 overflow-y-auto px-4 py-6 pb-32">
            <div class="max-w-4xl mx-auto space-y-6">
                <!-- 欢迎界面（没有消息时显示） -->
                <div v-if="messages.length === 0" class="text-center py-20">
                    <div
                        class="w-20 h-20 bg-gradient-to-r from-purple-500 to-blue-500 rounded-2xl flex items-center justify-center mx-auto mb-6">
                        <i class="fas fa-robot text-white text-4xl"></i>
                    </div>
                    <h2 class="text-3xl font-bold text-gray-900 mb-3">How can I help you today?</h2>
                    <p class="text-gray-600 mb-8">Ask me anything about chemistry, drug design, or ADMET properties</p>

                    <!-- 示例问题 -->
                    <div class="grid md:grid-cols-2 gap-4 max-w-2xl mx-auto">
                        <button v-for="example in exampleQuestions" :key="example" @click="handleExampleClick(example)"
                            class="p-4 bg-white border border-gray-200 rounded-xl hover:border-blue-500 hover:shadow-md transition-all text-left group">
                            <i class="fas fa-lightbulb text-yellow-500 mr-2"></i>
                            <span class="text-gray-700 group-hover:text-blue-600">{{ example }}</span>
                        </button>
                    </div>
                </div>

                <!-- 消息列表 -->
                <div v-for="(msg, index) in messages" :key="index" class="flex"
                    :class="msg.role === 'user' ? 'justify-end' : 'justify-start'">
                    <!-- AI 消息 -->
                    <div v-if="msg.role === 'assistant'" class="flex space-x-3 max-w-3xl">
                        <div
                            class="w-8 h-8 bg-gradient-to-r from-purple-500 to-blue-500 rounded-lg flex-shrink-0 flex items-center justify-center">
                            <i class="fas fa-robot text-white text-sm"></i>
                        </div>
                        <div class="flex-1 bg-white rounded-2xl px-5 py-4  border border-gray-100">
                            <!-- 加载状态（消息为空时显示） -->
                            <div v-if="!msg.content" class="flex items-center space-x-2">
                                <div class="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                                    style="animation-delay: 0s">
                                </div>
                                <div class="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                                    style="animation-delay: 0.2s">
                                </div>
                                <div class="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                                    style="animation-delay: 0.4s">
                                </div>
                            </div>
                            <!-- 消息内容 -->
                            <div v-else>
                                <div class="prose prose-sm max-w-none" v-html="formatMessage(msg.content)"></div>
                                <div class="flex items-center space-x-2 mt-3 text-xs text-gray-400">
                                    <span>{{ formatTime(msg.timestamp) }}</span>
                                    <button @click="copyMessage(msg.content)" class="hover:text-gray-600">
                                        <i class="fas fa-copy"></i>
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- 用户消息 -->
                    <div v-else class="flex space-x-3 max-w-3xl">
                        <div class="flex-1 bg-blue-500 text-white rounded-2xl px-5 py-4 shadow-sm">
                            <div class="whitespace-pre-wrap">{{ msg.content }}</div>
                            <div class="text-xs text-blue-100 mt-2">{{ formatTime(msg.timestamp) }}</div>
                        </div>
                        <div class="w-8 h-8 bg-blue-600 rounded-lg flex-shrink-0 flex items-center justify-center">
                            <i class="fas fa-user text-white text-sm"></i>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- 输入区域（固定在底部） -->
        <div
            class="fixed bottom-0 left-0 right-0 px-4 py-4">
            <div class="max-w-4xl mx-auto  backdrop-blur-sm">
                <form @submit.prevent="handleSend" class="relative">
                    <textarea v-model="inputMessage" ref="textarea" :disabled="isLoading"
                        @keydown.enter.exact.prevent="handleSend" @input="autoResize"
                        placeholder="Type your message... (Press Enter to send, Shift+Enter for new line)"
                        class="w-full px-4 py-3 pr-24 border border-gray-300 rounded-xl resize-none focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none disabled:bg-gray-100 disabled:cursor-not-allowed shadow-sm"
                        rows="1" style="max-height: 200px"></textarea>

                    <div class="absolute right-5 bottom-3 flex items-center space-x-2">
                        <span class="text-xs text-gray-400">{{ inputMessage.length }} / 4000</span>
                        <button type="submit" :disabled="!inputMessage.trim() || isLoading"
                            class="p-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors shadow-sm">
                            <i v-if="isLoading" class="fas fa-spinner fa-spin"></i>
                            <i v-else class="fas fa-paper-plane"></i>
                        </button>
                    </div>
                </form>
             
            </div>
        </div>
    </div>
</template>

<script setup>
import { ref, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import message from '@/utils/message.js'
import { sendChatMessageStream } from '@/api/ai.js'

const router = useRouter()

// 状态
const messages = ref([])
const inputMessage = ref('')
const isLoading = ref(false)
const chatContainer = ref(null)
const textarea = ref(null)

// 示例问题
const exampleQuestions = [
    'What are the key ADMET properties to consider in drug design?',
    'Explain the Lipinski\'s Rule of Five',
    'How does bioavailability affect drug efficacy?',
    'What is the difference between in vitro and in vivo testing?'
]

// 自动调整 textarea 高度
const autoResize = () => {
    const el = textarea.value
    if (el) {
        el.style.height = 'auto'
        el.style.height = el.scrollHeight + 'px'
    }
}

// 滚动到底部
const scrollToBottom = () => {
    nextTick(() => {
        if (chatContainer.value) {
            chatContainer.value.scrollTop = chatContainer.value.scrollHeight
        }
    })
}

// 格式化消息（支持 Markdown）
const formatMessage = (content) => {
    // 简单的 Markdown 转换（可以后续使用 marked 或 markdown-it 库）
    return content
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/`(.*?)`/g, '<code class="bg-gray-100 px-1 rounded">$1</code>')
        .replace(/\n/g, '<br>')
        // h1~h6
        .replace(/^# (.*?)$/gm, '<h1>$1</h1>')
        .replace(/^## (.*?)$/gm, '<h2>$1</h2>')
        .replace(/^### (.*?)$/gm, '<h3>$1</h3>')
        .replace(/^#### (.*?)$/gm, '<h4>$1</h4>')
        .replace(/^##### (.*?)$/gm, '<h5>$1</h5>')
        .replace(/^###### (.*?)$/gm, '<h6>$1</h6>')
}

// 格式化时间
const formatTime = (timestamp) => {
    if (!timestamp) return ''
    const date = new Date(timestamp)
    const now = new Date()
    const diff = now - date

    if (diff < 60000) return 'Just now'
    if (diff < 3600000) return `${Math.floor(diff / 60000)} min ago`
    if (diff < 86400000) return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
}

// 复制消息
const copyMessage = (content) => {
    navigator.clipboard.writeText(content)
    message.success({
        title: 'Copied',
        message: 'Message copied to clipboard',
        duration: 2000
    })
}

// 处理示例问题点击
const handleExampleClick = (example) => {
    inputMessage.value = example
    nextTick(() => {
        autoResize()
    })
}

// 发送消息
const handleSend = async () => {
    const content = inputMessage.value.trim()
    if (!content || isLoading.value) return

    // 添加用户消息
    messages.value.push({
        role: 'user',
        content: content,
        timestamp: Date.now()
    })

    inputMessage.value = ''
    isLoading.value = true
    scrollToBottom()

    // 重置 textarea 高度
    nextTick(() => {
        if (textarea.value) {
            textarea.value.style.height = 'auto'
        }
    })

    try {
        // 创建一个占位的 AI 消息
        const aiMessageIndex = messages.value.length
        messages.value.push({
            role: 'assistant',
            content: '',
            timestamp: Date.now()
        })
        scrollToBottom()

        // 调用流式 API（使用后端默认配置）
        await sendChatMessageStream(
            {
                messages: messages.value
                    .filter((msg, idx) => idx !== aiMessageIndex) // 排除占位消息
                    .map(msg => ({
                        role: msg.role,
                        content: msg.content
                    }))
            },
            // onChunk: 接收到数据块
            (chunk) => {
                messages.value[aiMessageIndex].content += chunk
                scrollToBottom()
            },
            // onError: 错误处理
            (error) => {
                console.error('Send message failed:', error)

                // 移除占位消息
                messages.value.splice(aiMessageIndex, 1)

                message.error({
                    title: 'Error',
                    message: error.message || 'Failed to send message',
                    duration: 3000
                })

                isLoading.value = false
            },
            // onComplete: 完成
            () => {
                isLoading.value = false
                scrollToBottom()
            }
        )
    } catch (error) {
        console.error('Send message failed:', error)
        message.error({
            title: 'Error',
            message: 'Failed to send message',
            duration: 3000
        })
        isLoading.value = false
    }
}
</script>

<style scoped>
/* 自定义滚动条 */
.overflow-y-auto::-webkit-scrollbar {
    width: 6px;
}

.overflow-y-auto::-webkit-scrollbar-track {
    background: #f1f1f1;
}

.overflow-y-auto::-webkit-scrollbar-thumb {
    background: #888;
    border-radius: 3px;
}

.overflow-y-auto::-webkit-scrollbar-thumb:hover {
    background: #555;
}

/* 打字机动画 */
@keyframes bounce {

    0%,
    100% {
        transform: translateY(0);
    }

    50% {
        transform: translateY(-4px);
    }
}

/* Prose 样式（Markdown 渲染） */
.prose {
    color: #374151;
    line-height: 1.75;
}

.prose strong {
    color: #111827;
    font-weight: 600;
}

.prose em {
    color: #374151;
    font-style: italic;
}

.prose code {
    color: #e11d48;
    font-family: 'Courier New', monospace;
    font-size: 0.875em;
}
</style>
