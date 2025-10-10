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
                    <router-link to="/optimization"
                        class="text-gray-300 hover:text-white px-4 py-2 font-medium transition duration-300 hover:scale-105">
                        <i class="fas fa-cogs mr-2 hover:-rotate-12 transition-transform duration-300"></i>Optimization
                    </router-link>
                    <a href="#"
                        class="text-gray-300 hover:text-white px-4 py-2 font-medium transition duration-300 hover:scale-105">
                        <i class="fas fa-question-circle mr-2 hover:bounce"></i>Help
                    </a>
                    <a href="#"
                        class="text-gray-300 hover:text-white px-4 py-2 font-medium transition duration-300 hover:scale-105">
                        <i class="fas fa-envelope mr-2 hover:wiggle"></i>Contact
                    </a>
                    <span
                        class="text-gray-400 text-sm border border-gray-600 px-2 py-1 rounded hover:border-gray-500 transition-colors duration-300">
                        <i class="fas fa-users text-xs mr-1"></i>CBDD GROUP
                    </span>
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
import { ref } from 'vue'

const isMobileMenuOpen = ref(false)

const toggleMobileMenu = () => {
    isMobileMenuOpen.value = !isMobileMenuOpen.value
}

const closeMobileMenu = () => {
    isMobileMenuOpen.value = false
}
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