<template>
    <div
        class="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 flex items-center justify-center p-4">
        <div class="max-w-md w-full">
            <!-- 登录卡片 -->
            <div class="bg-white rounded-2xl  overflow-hidden border border-gray-100">
                <!-- 头部装饰 -->
                <div class="relative bg-blue-600 p-4">
                    <div class="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent"></div>
                    <div class="relative text-center">
                        <div
                            class="inline-flex items-center justify-center w-16 h-16 bg-white/20 rounded-2xl mb-4 backdrop-blur-sm">
                            <i class="fas fa-user-circle text-2xl text-white"></i>
                        </div>
                        <h2 class="text-2xl font-bold text-white mb-2">Welcome Back</h2>
                        <p class="text-blue-100 text-sm">Sign in to your ADMETchem account</p>
                    </div>
                </div>

                <!-- 登录表单 -->
                <div class="p-8">
                    <form @submit.prevent="handleLogin" class="space-y-6">
                        <!-- Email Input -->
                        <div>
                            <label class="block text-sm font-semibold text-gray-700 mb-2">
                                <i class="fas fa-envelope text-blue-500 mr-2"></i>Email
                            </label>
                            <input v-model="formData.email" type="email" required placeholder="Enter your email"
                                class="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200 outline-none"
                                :disabled="loading" />
                        </div>

                        <!-- Password Input -->
                        <div>
                            <label class="block text-sm font-semibold text-gray-700 mb-2">
                                <i class="fas fa-lock text-blue-500 mr-2"></i>Password
                            </label>
                            <div class="relative">
                                <input v-model="formData.password" :type="showPassword ? 'text' : 'password'" required
                                    maxlength="20" placeholder="Enter your password"
                                    class="w-full px-4 py-3 pr-12 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200 outline-none"
                                    :disabled="loading" />
                                <button type="button" @click="showPassword = !showPassword"
                                    class="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors"
                                    tabindex="-1">
                                    <i :class="showPassword ? 'fas fa-eye-slash' : 'fas fa-eye'"></i>
                                </button>
                            </div>
                        </div>

                        <!-- Login Button -->
                        <button type="submit" :disabled="loading"
                            class="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-3 rounded-xl font-semibold hover:from-blue-700 hover:to-purple-700 focus:ring-4 focus:ring-blue-300 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2">
                            <i v-if="loading" class="fas fa-spinner fa-spin"></i>
                            <i v-else class="fas fa-sign-in-alt"></i>
                            <span>{{ loading ? 'Signing in...' : 'Sign In' }}</span>
                        </button>
                    </form>

                    <!-- Divider -->
                    <div class="relative my-6">
                        <div class="absolute inset-0 flex items-center">
                            <div class="w-full border-t border-gray-200"></div>
                        </div>
                        <div class="relative flex justify-center text-sm">
                            <span class="px-4 bg-white text-gray-500">Don't have an account?</span>
                        </div>
                    </div>

                    <!-- Register Link -->
                    <router-link to="/register"
                        class="block w-full text-center py-3 border-2 border-gray-300 text-gray-700 rounded-xl font-semibold hover:border-blue-500 hover:text-blue-600 transition-all duration-200">
                        <i class="fas fa-user-plus mr-2"></i>Create Account
                    </router-link>

                    <!-- Back to Home -->
                    <div class="mt-6 text-center">
                        <router-link to="/"
                            class="text-sm text-gray-600 hover:text-blue-600 transition-colors inline-flex items-center gap-1">
                            <i class="fas fa-arrow-left"></i>
                            <span>Back to Home</span>
                        </router-link>
                    </div>
                </div>
            </div>


        </div>
    </div>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { login } from '@/api/user.js'
import { setToken, setUser } from '@/utils/auth.js'
import message from '@/utils/message.js'

const router = useRouter()

const formData = ref({
    email: '',
    password: ''
})

const loading = ref(false)
const showPassword = ref(false)

const handleLogin = async () => {
    try {
        loading.value = true

        const response = await login({
            email: formData.value.email,
            password: formData.value.password
        })

        if (response.code === 0) {
            // 保存token和用户信息
            setToken(response.data.token.access_token)
            setUser(response.data.user)

            message.success({
                title: 'Login Successful',
                message: `Welcome back, ${response.data.user.username}!`,
                duration: 2000
            })

            // 跳转到首页或之前的页面
            setTimeout(() => {
                router.push('/')
            }, 1000)
        }
    } catch (error) {
        console.error('登录失败:', error)
        // 错误消息已在拦截器中处理
    } finally {
        loading.value = false
    }
}
</script>

<style scoped>
/* 自定义样式 */
input:disabled {
    background-color: #f9fafb;
    cursor: not-allowed;
}

/* 焦点样式 */
input:focus {
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

/* 按钮悬停效果 */
button:not(:disabled):hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
}

button:not(:disabled):active {
    transform: translateY(0);
}
</style>
