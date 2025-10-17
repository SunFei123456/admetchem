<template>
    <div
        class="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 flex items-center justify-center p-4">
        <div class="max-w-md w-full">
            <!-- 注册卡片 -->
            <div class="bg-white rounded-2xl  overflow-hidden border border-gray-100">
                <!-- 头部装饰 -->
                <div class="relative bg-blue-600 p-4">
                    <div class="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent"></div>
                    <div class="relative text-center">
                        <div
                            class="inline-flex items-center justify-center w-16 h-16 bg-white/20 rounded-2xl mb-4 backdrop-blur-sm">
                            <i class="fas fa-user-plus text-2xl text-white"></i>
                        </div>
                        <h2 class="text-2xl font-bold text-white mb-2">Create Account</h2>
                        <p class="text-blue-100 text-sm">Join ADMETchem and start your research journey</p>
                    </div>
                </div>

                <!-- 注册表单 -->
                <div class="p-8">
                    <form @submit.prevent="handleRegister" class="space-y-5">
                        <!-- Username Input -->
                        <div>
                            <label class="block text-sm font-semibold text-gray-700 mb-2">
                                <i class="fas fa-user text-blue-500 mr-2"></i>Username
                            </label>
                            <input v-model="formData.username" type="text" required minlength="2" maxlength="50"
                                placeholder="Enter username (2-50 characters)"
                                class="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200 outline-none"
                                :disabled="loading" />
                        </div>

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
                                    minlength="8" maxlength="20" placeholder="Enter password (8-20 characters)"
                                    class="w-full px-4 py-3 pr-12 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200 outline-none"
                                    :disabled="loading" />
                                <button type="button" @click="showPassword = !showPassword"
                                    class="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors"
                                    tabindex="-1">
                                    <i :class="showPassword ? 'fas fa-eye-slash' : 'fas fa-eye'"></i>
                                </button>
                            </div>
                        </div>

                        <!-- Confirm Password Input -->
                        <div>
                            <label class="block text-sm font-semibold text-gray-700 mb-2">
                                <i class="fas fa-lock text-blue-500 mr-2"></i>Confirm Password
                            </label>
                            <div class="relative">
                                <input v-model="formData.confirmPassword"
                                    :type="showConfirmPassword ? 'text' : 'password'" required minlength="8"
                                    maxlength="20" placeholder="Re-enter password"
                                    class="w-full px-4 py-3 pr-12 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200 outline-none"
                                    :disabled="loading" />
                                <button type="button" @click="showConfirmPassword = !showConfirmPassword"
                                    class="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors"
                                    tabindex="-1">
                                    <i :class="showConfirmPassword ? 'fas fa-eye-slash' : 'fas fa-eye'"></i>
                                </button>
                            </div>
                        </div>

                        <!-- Password Strength Indicator -->
                        <div v-if="formData.password" class="text-xs">
                            <div class="flex items-center gap-2 mb-1">
                                <span class="text-gray-600">Password Strength:</span>
                                <div class="flex gap-1 flex-1">
                                    <div v-for="i in 3" :key="i" class="h-1 flex-1 rounded-full transition-colors"
                                        :class="passwordStrength >= i ? getStrengthColor(passwordStrength) : 'bg-gray-200'">
                                    </div>
                                </div>
                                <span :class="getStrengthTextColor(passwordStrength)">{{
                                    getStrengthText(passwordStrength) }}</span>
                            </div>
                        </div>

                        <!-- Register Button -->
                        <button type="submit" :disabled="loading || !isFormValid"
                            class="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-3 rounded-xl font-semibold hover:from-blue-700 hover:to-purple-700 focus:ring-4 focus:ring-blue-300 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 mt-6">
                            <i v-if="loading" class="fas fa-spinner fa-spin"></i>
                            <i v-else class="fas fa-user-plus"></i>
                            <span>{{ loading ? 'Registering...' : 'Register' }}</span>
                        </button>
                    </form>

                    <!-- Divider -->
                    <div class="relative my-6">
                        <div class="absolute inset-0 flex items-center">
                            <div class="w-full border-t border-gray-200"></div>
                        </div>
                        <div class="relative flex justify-center text-sm">
                            <span class="px-4 bg-white text-gray-500">Already have an account?</span>
                        </div>
                    </div>

                    <!-- Login Link -->
                    <router-link to="/login"
                        class="block w-full text-center py-3 border-2 border-gray-300 text-gray-700 rounded-xl font-semibold hover:border-blue-500 hover:text-blue-600 transition-all duration-200">
                        <i class="fas fa-sign-in-alt mr-2"></i>Sign In Now
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
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { register } from '@/api/user.js'
import message from '@/utils/message.js'

const router = useRouter()

const formData = ref({
    username: '',
    email: '',
    password: '',
    confirmPassword: ''
})

const loading = ref(false)
const showPassword = ref(false)
const showConfirmPassword = ref(false)

// 计算密码强度
const passwordStrength = computed(() => {
    const password = formData.value.password
    if (!password) return 0

    let strength = 0
    if (password.length >= 8) strength++
    if (password.length >= 14) strength++
    if (/[a-z]/.test(password) && /[A-Z]/.test(password)) strength++
    if (/[0-9]/.test(password)) strength++
    if (/[^a-zA-Z0-9]/.test(password)) strength++

    return Math.min(3, Math.ceil(strength / 2))
})

// 表单验证
const isFormValid = computed(() => {
    return (
        formData.value.username.length >= 2 &&
        formData.value.email &&
        formData.value.password.length >= 8 &&
        formData.value.password.length <= 20 &&
        formData.value.password === formData.value.confirmPassword
    )
})

const getStrengthColor = (strength) => {
    if (strength === 1) return 'bg-red-500'
    if (strength === 2) return 'bg-yellow-500'
    return 'bg-green-500'
}

const getStrengthTextColor = (strength) => {
    if (strength === 1) return 'text-red-600'
    if (strength === 2) return 'text-yellow-600'
    return 'text-green-600'
}

const getStrengthText = (strength) => {
    if (strength === 1) return 'Weak'
    if (strength === 2) return 'Medium'
    return 'Strong'
}

const handleRegister = async () => {
    // Validate password length
    if (formData.value.password.length > 20) {
        message.error({
            title: 'Registration Failed',
            message: 'Password must be no longer than 20 characters',
            duration: 3000
        })
        return
    }

    // Validate password match
    if (formData.value.password !== formData.value.confirmPassword) {
        message.error({
            title: 'Registration Failed',
            message: 'Passwords do not match',
            duration: 3000
        })
        return
    }

    try {
        loading.value = true

        const response = await register({
            username: formData.value.username,
            email: formData.value.email,
            password: formData.value.password
        })

        if (response.code === 0) {
            message.success({
                title: 'Registration Successful',
                message: 'Redirecting to login page...',
                duration: 2000
            })

            // Redirect to login page
            setTimeout(() => {
                router.push('/login')
            }, 1500)
        }
    } catch (error) {
        console.error('Registration failed:', error)
        // Error message handled by interceptor
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
