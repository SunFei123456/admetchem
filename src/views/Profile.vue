<template>
    <div class="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 py-12 px-4">
        <div class="max-w-4xl mx-auto">
            <!-- Page Title -->
            <div class="mb-8">
                <h1
                    class="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-2">
                    Profile
                </h1>
                <p class="text-gray-600">Manage your account information and preferences</p>
            </div>

            <!-- Loading State -->
            <div v-if="loading" class="flex justify-center items-center py-20">
                <div class="text-center">
                    <i class="fas fa-spinner fa-spin text-4xl text-blue-500 mb-4"></i>
                    <p class="text-gray-600">Loading...</p>
                </div>
            </div>

            <!-- 内容区域 -->
            <div v-else class="space-y-6">
                <!-- 用户信息卡片 -->
                <div class="bg-white rounded-3xl shadow-xl border border-gray-100 overflow-hidden">
                    <!-- 头部装饰 -->
                    <div class="relative bg-gradient-to-r from-blue-600 via-purple-600 to-blue-700 p-8">
                        <div class="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent">
                        </div>
                        <div class="relative flex items-center">
                            <!-- 用户头像 -->
                            <div v-if="user.avatar"
                                class="w-20 h-20 rounded-2xl overflow-hidden border-4 border-white/30 shadow-lg">
                                <img :src="user.avatar" :alt="user.username" class="w-full h-full object-cover" />
                            </div>
                            <div v-else
                                class="w-20 h-20 bg-white/20 rounded-2xl flex items-center justify-center backdrop-blur-sm">
                                <i class="fas fa-user text-4xl text-white"></i>
                            </div>
                            <div class="ml-6">
                                <h2 class="text-3xl font-bold text-white">{{ user.username }}</h2>
                                <p class="text-blue-100 mt-1">{{ user.email }}</p>
                            </div>
                        </div>
                    </div>

                    <!-- User Information Details -->
                    <div class="p-8">
                        <div class="grid md:grid-cols-2 gap-6">
                            <div class="space-y-2">
                                <label class="text-sm font-semibold text-gray-500 uppercase tracking-wide">User
                                    ID</label>
                                <p class="text-lg text-gray-900">{{ user.id }}</p>
                            </div>
                            <div class="space-y-2">
                                <label class="text-sm font-semibold text-gray-500 uppercase tracking-wide">Account
                                    Status</label>
                                <span class="inline-flex items-center px-3 py-1 rounded-full text-sm font-semibold"
                                    :class="user.is_active ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'">
                                    <i :class="user.is_active ? 'fas fa-check-circle' : 'fas fa-times-circle'"
                                        class="mr-2"></i>
                                    {{ user.is_active ? 'Active' : 'Inactive' }}
                                </span>
                            </div>
                            <div class="space-y-2">
                                <label class="text-sm font-semibold text-gray-500 uppercase tracking-wide">Created
                                    At</label>
                                <p class="text-lg text-gray-900">{{ formatDate(user.created_at) }}</p>
                            </div>
                            <div class="space-y-2">
                                <label class="text-sm font-semibold text-gray-500 uppercase tracking-wide">Last
                                    Updated</label>
                                <p class="text-lg text-gray-900">{{ formatDate(user.updated_at || user.created_at) }}
                                </p>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Edit Information Card -->
                <div class="bg-white rounded-3xl shadow-xl border border-gray-100 overflow-hidden">
                    <div class="border-b border-gray-100 p-6">
                        <h3 class="text-xl font-bold text-gray-900 flex items-center">
                            <i class="fas fa-edit text-blue-500 mr-3"></i>
                            Edit Information
                        </h3>
                    </div>

                    <div class="p-8">
                        <form @submit.prevent="handleUpdate" class="space-y-6">
                            <!-- Edit Username -->
                            <div>
                                <label class="block text-sm font-semibold text-gray-700 mb-2">
                                    <i class="fas fa-user text-blue-500 mr-2"></i>Username
                                </label>
                                <input v-model="editForm.username" type="text" minlength="2" maxlength="50"
                                    placeholder="Enter new username"
                                    class="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200 outline-none"
                                    :disabled="updating" />
                                <p class="mt-2 text-sm text-gray-500">
                                    <i class="fas fa-info-circle mr-1"></i>Current username: {{ user.username }}
                                </p>
                            </div>

                            <!-- Edit Avatar -->
                            <div>
                                <label class="block text-sm font-semibold text-gray-700 mb-2">
                                    <i class="fas fa-image text-blue-500 mr-2"></i>Avatar
                                </label>

                                <!-- 上传按钮和输入框组合 -->
                                <div class="flex gap-3">
                                    <!-- Avatar URL 输入框 -->
                                    <input v-model="editForm.avatar" type="url"
                                        placeholder="Enter avatar URL or upload an image"
                                        class="flex-1 px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200 outline-none"
                                        :disabled="updating || uploading" />

                                    <!-- 上传按钮 -->
                                    <button type="button" @click="handleUploadAvatar" :disabled="updating || uploading"
                                        class="px-6 py-3 bg-gradient-to-r from-green-500 to-green-600 text-white rounded-xl font-semibold hover:from-green-600 hover:to-green-700 focus:ring-4 focus:ring-green-300 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 whitespace-nowrap">
                                        <i v-if="uploading" class="fas fa-spinner fa-spin"></i>
                                        <i v-else class="fas fa-cloud-upload-alt"></i>
                                        <span>{{ uploading ? 'Uploading...' : 'Upload' }}</span>
                                    </button>
                                </div>

                                <div class="mt-3 flex items-center justify-between">
                                    <p class="text-sm text-gray-500">
                                        <i class="fas fa-info-circle mr-1"></i>Upload or enter URL (jpg, png, gif, webp
                                        - max 3MB)
                                    </p>
                                    <!-- Avatar Preview -->
                                    <div v-if="editForm.avatar" class="flex items-center gap-2">
                                        <span class="text-sm text-gray-500">Preview:</span>
                                        <div
                                            class="w-12 h-12 rounded-full overflow-hidden border-2 border-green-400 shadow-md">
                                            <img :src="editForm.avatar" alt="Avatar preview"
                                                class="w-full h-full object-cover" />
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <!-- Change Password Section -->
                            <div class="border-t border-gray-200 pt-6">
                                <h4 class="text-lg font-semibold text-gray-900 mb-4">Change Password</h4>

                                <!-- Current Password -->
                                <div class="mb-4">
                                    <label class="block text-sm font-semibold text-gray-700 mb-2">
                                        <i class="fas fa-lock text-blue-500 mr-2"></i>Current Password
                                    </label>
                                    <div class="relative">
                                        <input v-model="editForm.old_password"
                                            :type="showOldPassword ? 'text' : 'password'"
                                            placeholder="Enter current password to change"
                                            class="w-full px-4 py-3 pr-12 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200 outline-none"
                                            :disabled="updating" />
                                        <button type="button" @click="showOldPassword = !showOldPassword"
                                            class="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                                            tabindex="-1">
                                            <i :class="showOldPassword ? 'fas fa-eye-slash' : 'fas fa-eye'"></i>
                                        </button>
                                    </div>
                                </div>

                                <!-- New Password -->
                                <div class="mb-4">
                                    <label class="block text-sm font-semibold text-gray-700 mb-2">
                                        <i class="fas fa-key text-blue-500 mr-2"></i>New Password
                                    </label>
                                    <div class="relative">
                                        <input v-model="editForm.new_password"
                                            :type="showNewPassword ? 'text' : 'password'" minlength="8" maxlength="20"
                                            placeholder="Enter new password (8-20 characters)"
                                            class="w-full px-4 py-3 pr-12 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200 outline-none"
                                            :disabled="updating || !editForm.old_password" />
                                        <button type="button" @click="showNewPassword = !showNewPassword"
                                            class="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                                            tabindex="-1">
                                            <i :class="showNewPassword ? 'fas fa-eye-slash' : 'fas fa-eye'"></i>
                                        </button>
                                    </div>
                                </div>

                                <!-- Confirm New Password -->
                                <div>
                                    <label class="block text-sm font-semibold text-gray-700 mb-2">
                                        <i class="fas fa-check-circle text-blue-500 mr-2"></i>Confirm New Password
                                    </label>
                                    <div class="relative">
                                        <input v-model="editForm.confirm_password"
                                            :type="showConfirmPassword ? 'text' : 'password'" minlength="8"
                                            maxlength="20" placeholder="Re-enter new password"
                                            class="w-full px-4 py-3 pr-12 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200 outline-none"
                                            :disabled="updating || !editForm.new_password" />
                                        <button type="button" @click="showConfirmPassword = !showConfirmPassword"
                                            class="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                                            tabindex="-1">
                                            <i :class="showConfirmPassword ? 'fas fa-eye-slash' : 'fas fa-eye'"></i>
                                        </button>
                                    </div>
                                </div>
                            </div>

                            <!-- Submit Buttons -->
                            <div class="flex gap-4 pt-4">
                                <button type="submit" :disabled="updating || !hasChanges"
                                    class="flex-1 bg-gradient-to-r from-blue-600 to-purple-600 text-white py-3 rounded-xl font-semibold hover:from-blue-700 hover:to-purple-700 focus:ring-4 focus:ring-blue-300 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2">
                                    <i v-if="updating" class="fas fa-spinner fa-spin"></i>
                                    <i v-else class="fas fa-save"></i>
                                    <span>{{ updating ? 'Saving...' : 'Save Changes' }}</span>
                                </button>
                                <button type="button" @click="resetForm" :disabled="updating"
                                    class="px-6 py-3 border-2 border-gray-300 text-gray-700 rounded-xl font-semibold hover:border-gray-400 hover:bg-gray-50 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed">
                                    <i class="fas fa-undo mr-2"></i>Reset
                                </button>
                            </div>
                        </form>
                    </div>
                </div>

                <!-- Logout Card -->
                <div class="bg-white rounded-3xl shadow-xl border border-gray-100 overflow-hidden">
                    <div class="p-6">
                        <div class="flex items-center justify-between">
                            <div>
                                <h3 class="text-lg font-bold text-gray-900 mb-1">Sign Out</h3>
                                <p class="text-sm text-gray-600">Log out of your account</p>
                            </div>
                            <button @click="handleLogout"
                                class="px-6 py-3 bg-gradient-to-r from-red-500 to-red-600 text-white rounded-xl font-semibold hover:from-red-600 hover:to-red-700 focus:ring-4 focus:ring-red-300 transition-all duration-200 flex items-center gap-2">
                                <i class="fas fa-sign-out-alt"></i>
                                <span>Sign Out</span>
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { getCurrentUser, updateCurrentUser } from '@/api/user.js'
import { clearAuth, setUser } from '@/utils/auth.js'
import message from '@/utils/message.js'
import { loadCloudinaryScript, openUploadWidget } from '@/utils/cloudinary.js'

const router = useRouter()

const user = ref({})
const loading = ref(true)
const updating = ref(false)
const uploading = ref(false) // 图片上传状态

const editForm = ref({
    username: '',
    avatar: '',
    old_password: '',
    new_password: '',
    confirm_password: ''
})

const showOldPassword = ref(false)
const showNewPassword = ref(false)
const showConfirmPassword = ref(false)

// 检查是否有修改
const hasChanges = computed(() => {
    const usernameChanged = editForm.value.username && editForm.value.username !== user.value.username
    const avatarChanged = editForm.value.avatar !== undefined && editForm.value.avatar !== user.value.avatar
    const passwordChanged = editForm.value.old_password && editForm.value.new_password
    return usernameChanged || avatarChanged || passwordChanged
})

// 格式化日期
const formatDate = (dateString) => {
    if (!dateString) return '-'
    const date = new Date(dateString)
    return date.toLocaleString('zh-CN', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit'
    })
}

// 加载用户信息
const loadUserInfo = async () => {
    try {
        loading.value = true
        const response = await getCurrentUser()

        if (response.code === 0) {
            user.value = response.data
            editForm.value.username = response.data.username
            editForm.value.avatar = response.data.avatar || ''
        }
    } catch (error) {
        console.error('Failed to load user info:', error)
        message.error({
            title: 'Load Failed',
            message: 'Unable to get user information, please login again',
            duration: 3000
        })
        // If loading fails, token may be expired, redirect to login
        setTimeout(() => {
            handleLogout()
        }, 2000)
    } finally {
        loading.value = false
    }
}

// Update user information
const handleUpdate = async () => {
    // Validate changes
    if (editForm.value.new_password) {
        if (!editForm.value.old_password) {
            message.error({
                title: 'Validation Failed',
                message: 'Please enter current password',
                duration: 3000
            })
            return
        }

        if (editForm.value.new_password !== editForm.value.confirm_password) {
            message.error({
                title: 'Validation Failed',
                message: 'New passwords do not match',
                duration: 3000
            })
            return
        }

        if (editForm.value.new_password.length < 8) {
            message.error({
                title: 'Validation Failed',
                message: 'New password must be at least 8 characters',
                duration: 3000
            })
            return
        }

        if (editForm.value.new_password.length > 20) {
            message.error({
                title: 'Validation Failed',
                message: 'New password must be no longer than 20 characters',
                duration: 3000
            })
            return
        }
    }

    try {
        updating.value = true

        const updateData = {}

        // 只发送有修改的字段
        if (editForm.value.username && editForm.value.username !== user.value.username) {
            updateData.username = editForm.value.username
        }

        if (editForm.value.avatar !== undefined && editForm.value.avatar !== user.value.avatar) {
            updateData.avatar = editForm.value.avatar
        }

        if (editForm.value.old_password && editForm.value.new_password) {
            updateData.old_password = editForm.value.old_password
            updateData.new_password = editForm.value.new_password
        }

        const response = await updateCurrentUser(updateData)

        if (response.code === 0) {
            user.value = response.data
            setUser(response.data)

            message.success({
                title: 'Update Successful',
                message: 'Your information has been updated',
                duration: 2000
            })

            // Reset form
            resetForm()
        }
    } catch (error) {
        console.error('Update failed:', error)
        // Error message handled by interceptor
    } finally {
        updating.value = false
    }
}

// Reset form
const resetForm = () => {
    editForm.value = {
        username: user.value.username,
        avatar: user.value.avatar || '',
        old_password: '',
        new_password: '',
        confirm_password: ''
    }
}

// Logout
const handleLogout = () => {
    clearAuth()
    message.success({
        title: 'Signed Out',
        message: 'You have successfully logged out',
        duration: 2000
    })
    setTimeout(() => {
        router.push('/')
    }, 1000)
}

// 上传头像到 Cloudinary
const handleUploadAvatar = async () => {
    try {
        uploading.value = true

        // 打开 Cloudinary 上传 Widget
        openUploadWidget(
            {
                // 可以自定义配置
                maxFileSize: 3000000, // 3MB
                croppingAspectRatio: 1, // 正方形
            },
            // 成功回调
            (url, publicId) => {
                editForm.value.avatar = url
                uploading.value = false
                message.success({
                    title: 'Upload Successful',
                    message: 'Avatar uploaded successfully!',
                    duration: 2000
                })
            },
            // 失败回调
            (error) => {
                uploading.value = false
                console.error('Upload failed:', error)
                message.error({
                    title: 'Upload Failed',
                    message: error.message || 'Failed to upload image',
                    duration: 3000
                })
            }
        )
    } catch (error) {
        uploading.value = false
        console.error('Error opening upload widget:', error)
        message.error({
            title: 'Error',
            message: 'Failed to open upload widget',
            duration: 3000
        })
    }
}

// Load user info on page mount
onMounted(async () => {
    // 加载 Cloudinary 脚本
    try {
        await loadCloudinaryScript()
        console.log('✅ Cloudinary script loaded')
    } catch (error) {
        console.error('❌ Failed to load Cloudinary script:', error)
    }

    // 加载用户信息
    loadUserInfo()
})
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
}

button:not(:disabled):active {
    transform: translateY(0);
}
</style>
