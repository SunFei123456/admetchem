<template>
    <div class="bg-gradient-to-r from-blue-100 via-purple-50 to-pink-100 border-b border-gray-200 py-3">
        <div class="max-w-7xl mx-auto px-4">
            <div class="flex items-center justify-between">
                <!-- 面包屑导航 -->
                <nav class="flex items-center space-x-2 text-sm">
                    <router-link to="/"
                        class="flex items-center text-gray-600 hover:text-blue-600 transition duration-200">
                        <i class="fas fa-home mr-1 hover:animate-bounce"></i>
                        Home
                    </router-link>

                    <template v-if="currentRoute.name !== 'Home'">
                        <i class="fas fa-chevron-right text-gray-400 text-xs"></i>

                        <template v-if="currentRoute.name === 'DruglikenessEvaluation'">
                            <span class="text-gray-600">Webserver</span>
                            <i class="fas fa-chevron-right text-gray-400 text-xs"></i>
                            <span class="text-gray-800 font-bold">Druglikeness Evaluation</span>
                        </template>

                        <template v-else-if="currentRoute.name === 'ADMETPrediction'">
                            <span class="text-gray-600">Webserver</span>
                            <i class="fas fa-chevron-right text-gray-400 text-xs"></i>
                            <span class="text-gray-800 font-bold">ADMET Prediction</span>
                        </template>

                        <template v-else-if="currentRoute.name === 'SystematicEvaluation'">
                            <span class="text-gray-600">Webserver</span>
                            <i class="fas fa-chevron-right text-gray-400 text-xs"></i>
                            <span class="text-gray-800 font-bold">Systematic Evaluation</span>
                        </template>

                        <template v-else-if="currentRoute.name === 'ApplicationDomain'">
                            <span class="text-gray-600">Webserver</span>
                            <i class="fas fa-chevron-right text-gray-400 text-xs"></i>
                            <span class="text-gray-800 font-bold">Application Domain</span>
                        </template>

                        <template v-else-if="currentRoute.name === 'About'">
                            <span class="text-gray-800 font-bold">About</span>
                        </template>

                        <template v-else-if="currentRoute.name === 'Search'">
                            <span class="text-gray-800 font-bold">Search</span>
                        </template>

                        <template v-else-if="currentRoute.name === 'DruglikenessResult'">
                            <router-link to="/druglikeness-evaluation" class="text-gray-600 hover:text-blue-600 transition duration-200">Webserver</router-link>
                            <i class="fas fa-chevron-right text-gray-400 text-xs"></i>
                          
                            <span class="text-gray-800 font-bold">Druglikeness Evaluation result</span>
                        </template>

                        <template v-else>
                            <span class="text-gray-800 font-bold">{{ currentRoute.name }}</span>
                        </template>
                    </template>
                </nav>

                <!-- 实时状态信息 -->
                <div class="flex items-center space-x-6 text-sm">
                    <!-- 在线状态 -->
                    <div class="flex items-center space-x-2">
                        <div class="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                        <span class="text-gray-600">系统在线</span>
                    </div>

                    <!-- 当前时间 -->
                    <div class="flex items-center space-x-2">
                        <i class="fas fa-clock text-blue-500 hover:animate-spin"></i>
                        <span class="text-gray-600">{{ currentTime }}</span>
                    </div>

                </div>
            </div>
        </div>
    </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, computed } from 'vue'
import { useRoute } from 'vue-router'

const route = useRoute()
const currentTime = ref('')
let timeInterval = null

// 获取当前路由信息
const currentRoute = computed(() => route)

// 更新时间
const updateTime = () => {
    const now = new Date()
    currentTime.value = now.toLocaleTimeString('zh-CN', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    })
}

onMounted(() => {
    updateTime()
    timeInterval = setInterval(updateTime, 1000)
})

onUnmounted(() => {
    if (timeInterval) {
        clearInterval(timeInterval)
    }
})
</script>

<style scoped>
/* 动画效果 */
@keyframes wiggle {

    0%,
    100% {
        transform: rotate(0deg);
    }

    25% {
        transform: rotate(-3deg);
    }

    75% {
        transform: rotate(3deg);
    }
}

.hover\:animate-wiggle:hover {
    animation: wiggle 0.3s ease-in-out;
}
</style>