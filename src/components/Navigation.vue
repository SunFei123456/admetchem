<template>
    <nav class="bg-gray-800 sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-4">
            <div class="flex justify-between items-center py-3">
                <!-- Logo -->
                <div class="flex items-center space-x-3">
                    <div
                        class="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center hover:bg-green-600 transition-all duration-300 hover:scale-110 logo-glow">
                        <i class="fas fa-flask text-white text-sm hover:animate-pulse"></i>
                    </div>
                    <span class="text-white text-xl font-bold">ADMET<span class="text-green-400">chem</span></span>
                </div>

                <!-- 主导航 -->
                <div class="hidden md:flex items-center space-x-6">
                    <router-link to="/"
                        class="bg-red-600 text-white px-4 py-2 rounded font-medium hover:bg-red-700 transition duration-300 hover:scale-105"
                        v-if="$route.path === '/'">
                        <i class="fas fa-home mr-2 hover:bounce"></i>Home
                    </router-link>
                    <router-link to="/"
                        class="text-gray-300 hover:text-white px-4 py-2 font-medium transition duration-300 hover:scale-105"
                        v-else>
                        <i class="fas fa-home mr-2 hover:bounce"></i>Home
                    </router-link>

                    <div class="relative group">
                        <button
                            class="text-gray-300 hover:text-white px-4 py-2 font-medium transition duration-300 flex items-center">
                            <i class="fas fa-server mr-2"></i>Webserver <i
                                class="fas fa-chevron-down ml-1 transition-transform duration-300 group-hover:rotate-180"></i>
                        </button>

                        <!-- 二级下拉菜单 -->
                        <div
                            class="absolute left-0 top-full mt-1 w-64 bg-white border border-gray-200 rounded-lg shadow-lg opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-300 transform translate-y-2 group-hover:translate-y-0 z-50">
                            <div class="py-2">
                                <router-link to="/druglikeness-evaluation"
                                    class="flex items-center px-4 py-3 text-gray-700 hover:bg-blue-50 hover:text-blue-600 transition duration-200">
                                    <i class="fas fa-pills text-blue-500 mr-3 w-4 text-center"></i>
                                    <span class="font-medium">Druglikeness Evaluation</span>
                                </router-link>
                                <router-link to="/admet-prediction"
                                    class="flex items-center px-4 py-3 text-gray-700 hover:bg-green-50 hover:text-green-600 transition duration-200">
                                    <i class="fas fa-brain text-green-500 mr-3 w-4 text-center"></i>
                                    <span class="font-medium">ADMET Prediction</span>
                                </router-link>
                            </div>
                        </div>
                    </div>

                    <router-link to="/search"
                        class="text-gray-300 hover:text-white px-4 py-2 font-medium transition duration-300 hover:scale-105">
                        <i class="fas fa-search mr-2 hover:rotate-90 transition-transform duration-300"></i>Search
                    </router-link>
                    <router-link to="/ai-chat"
                        class="text-gray-300 hover:text-white px-4 py-2 font-medium transition duration-300 hover:scale-105">
                        <i class="fas fa-robot mr-2 hover:bounce transition-transform duration-300"></i>AI Chat
                    </router-link>
                    <a href="#"
                        class="text-gray-300 hover:text-white px-4 py-2 font-medium transition duration-300 hover:scale-105">
                        <i class="fas fa-question-circle mr-2 hover:bounce"></i>Help
                    </a>

                    <!-- 用户认证区域 -->
                    <div class="flex items-center space-x-2 ml-4 pl-4 border-l border-gray-600">
                        <!-- 未登录 -->
                        <template v-if="!isLoggedIn">
                            <router-link to="/login"
                                class="text-gray-300 hover:text-white px-3 py-1.5 font-medium transition duration-300 hover:scale-105">
                                <i class="fas fa-sign-in-alt mr-1"></i>Login
                            </router-link>
                            <router-link to="/register"
                                class="bg-blue-600 text-white px-3 py-1.5 rounded font-medium hover:bg-blue-700 transition duration-300 hover:scale-105">
                                <i class="fas fa-user-plus mr-1"></i>Register
                            </router-link>
                        </template>

                        <!-- 已登录 -->
                        <div v-else class="relative group">
                            <button
                                class="flex items-center space-x-2 text-gray-300 hover:text-white px-3 py-1.5 font-medium transition duration-300">
                                <!-- 用户头像 -->
                                <div v-if="currentUser?.avatar"
                                    class="w-8 h-8 rounded-full overflow-hidden border-2 border-blue-400">
                                    <img :src="currentUser.avatar" :alt="currentUser.username"
                                        class="w-full h-full object-cover" />
                                </div>
                                <div v-else class="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center">
                                    <i class="fas fa-user text-white text-sm"></i>
                                </div>
                                <span>{{ currentUser?.username || 'User' }}</span>
                                <i
                                    class="fas fa-chevron-down text-xs transition-transform duration-300 group-hover:rotate-180"></i>
                            </button>

                            <!-- 用户下拉菜单 -->
                            <div
                                class="absolute right-0 top-full mt-1 w-48 bg-white border border-gray-200 rounded-lg shadow-lg opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-300 transform translate-y-2 group-hover:translate-y-0 z-50">
                                <div class="py-2">
                                    <router-link to="/profile"
                                        class="flex items-center px-4 py-2 text-gray-700 hover:bg-blue-50 hover:text-blue-600 transition duration-200">
                                        <i class="fas fa-user-circle text-blue-500 mr-3"></i>
                                        <span>Profile</span>
                                    </router-link>
                                    <div class="border-t border-gray-200 my-1"></div>
                                    <button @click="handleLogout"
                                        class="w-full flex items-center px-4 py-2 text-gray-700 hover:bg-red-50 hover:text-red-600 transition duration-200">
                                        <i class="fas fa-sign-out-alt text-red-500 mr-3"></i>
                                        <span>Logout</span>
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 移动端菜单按钮 -->
                <button @click="toggleMobileMenu"
                    class="md:hidden text-gray-300 hover:text-white transition duration-300">
                    <i class="fas fa-bars text-xl"></i>
                </button>
            </div>



            <!-- 移动端菜单 -->
            <div v-if="isMobileMenuOpen" class="md:hidden py-4">
                <router-link to="/" @click="closeMobileMenu"
                    class="block py-2 text-gray-300 hover:text-white transition duration-300">
                    <i class="fas fa-home mr-2"></i>Home
                </router-link>
                <a href="#" class="block py-2 text-gray-300 hover:text-white transition duration-300">
                    <i class="fas fa-server mr-2"></i>Webserver
                </a>
                <router-link to="/search" @click="closeMobileMenu"
                    class="block py-2 text-gray-300 hover:text-white transition duration-300">
                    <i class="fas fa-search mr-2"></i>Search
                </router-link>
            </div>
        </div>
    </nav>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { isLoggedIn as checkLoggedIn, getUser, clearAuth } from '@/utils/auth.js'
import message from '@/utils/message.js'

const router = useRouter()
const isMobileMenuOpen = ref(false)
const currentUser = ref(null)
const isLoggedIn = ref(false) // 改为响应式 ref

// 加载用户信息
const loadUserInfo = () => {
    const loggedIn = checkLoggedIn()
    isLoggedIn.value = loggedIn // 更新登录状态

    if (loggedIn) {
        currentUser.value = getUser()
    } else {
        currentUser.value = null
    }
}

const toggleMobileMenu = () => {
    isMobileMenuOpen.value = !isMobileMenuOpen.value
}

const closeMobileMenu = () => {
    isMobileMenuOpen.value = false
}

// 退出登录
const handleLogout = () => {
    clearAuth()
    isLoggedIn.value = false // 更新登录状态
    currentUser.value = null
    message.success({
        title: '退出成功',
        message: '您已成功退出登录',
        duration: 2000
    })
    router.push('/')
}

// 组件挂载时加载用户信息
onMounted(() => {
    loadUserInfo()
})

// 监听路由变化，更新用户信息
router.afterEach(() => {
    loadUserInfo()
})
</script>

<style scoped>
/* 自定义动画 */
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

.hover\:wiggle:hover {
    animation: wiggle 0.3s ease-in-out;
}

/* 增强导航悬停效果 */
nav a:hover i {
    filter: drop-shadow(0 0 6px currentColor);
}

/* Logo 闪烁效果 */
.logo-glow:hover {
    box-shadow: 0 0 20px rgba(34, 197, 94, 0.5);
}
</style>