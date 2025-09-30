<template>
  <div class="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
    <!-- 现代化通知栏 -->
    <div class="relative overflow-hidden">
      <div class="absolute inset-0 bg-gradient-to-r from-blue-600 via-purple-600 to-blue-700 opacity-90"></div>
      <div class="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent animate-shimmer"></div>
      <div class="relative bg-gradient-to-r from-blue-600 via-purple-600 to-blue-700 p-6 shadow-lg">
        <div class="max-w-7xl mx-auto">
          <div class="flex items-center justify-center">
            <div class="flex items-center space-x-4">
              <div class="flex items-center">
                <div class="relative">
                  <span
                    class="bg-gradient-to-r from-orange-400 to-red-500 text-white px-4 py-2 rounded-full text-sm font-bold shadow-lg animate-pulse">
                    🎉 Good News
                  </span>
                  <div class="absolute -top-1 -right-1 w-3 h-3 bg-red-400 rounded-full animate-ping"></div>
                </div>
              </div>
              <div class="text-white font-medium text-lg">
                <span class="bg-white/20 px-3 py-1 rounded-full backdrop-blur-sm">
                  ADMETchem 2.0 is now online, welcome to have a try!
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="max-w-7xl mx-auto px-4 pb-12">
      <!-- 升级版滚动字幕 -->
      <section class="mt-8 mb-12">
        <div
          class="relative bg-gradient-to-r from-slate-800 via-slate-700 to-slate-800 rounded-2xl shadow-2xl overflow-hidden border border-slate-600">
          <div
            class="absolute inset-0 bg-gradient-to-r from-blue-500/10 via-purple-500/10 to-blue-500/10 animate-gradient">
          </div>
          <div class="relative p-6">
            <div class="flex items-center mb-2">
              <i class="fas fa-chart-line text-cyan-400 text-xl mr-3"></i>
              <h3 class="text-white font-bold text-lg">Performance Metrics</h3>
            </div>
            <div class="marquee-container">
              <div class="marquee-content">
                <p class="text-slate-300 font-medium text-base">
                  <strong class="text-cyan-300">Q² or ACC:</strong>
                  <span
                    class="metric-number bg-gradient-to-r from-green-400 to-emerald-500 bg-clip-text text-transparent font-bold">18.</span>
                  Pgp-substrate: <span
                    class="metric-number bg-gradient-to-r from-blue-400 to-cyan-500 bg-clip-text text-transparent font-bold">0.824;</span>
                  HIA: <span
                    class="metric-number bg-gradient-to-r from-purple-400 to-pink-500 bg-clip-text text-transparent font-bold">0.782;</span>
                  F(20%): <span
                    class="metric-number bg-gradient-to-r from-orange-400 to-red-500 bg-clip-text text-transparent font-bold">0.689;</span>
                  F(30%): <span
                    class="metric-number bg-gradient-to-r from-indigo-400 to-purple-500 bg-clip-text text-transparent font-bold">0.669;</span>
                  hERG: <span
                    class="metric-number bg-gradient-to-r from-teal-400 to-blue-500 bg-clip-text text-transparent font-bold">0.844;</span>
                  H-HT: <span
                    class="metric-number bg-gradient-to-r from-rose-400 to-pink-500 bg-clip-text text-transparent font-bold">0.690;</span>
                  Ames: <span
                    class="metric-number bg-gradient-to-r from-amber-400 to-orange-500 bg-clip-text text-transparent font-bold">0.820;</span>
                  SkinSen: <span
                    class="metric-number bg-gradient-to-r from-lime-400 to-green-500 bg-clip-text text-transparent font-bold">0.706;</span>
                  CYP1A2-inhibitor: <span
                    class="metric-number bg-gradient-to-r from-violet-400 to-purple-500 bg-clip-text text-transparent font-bold">0.849;</span>
                  CYP1A2-substrate: <span
                    class="metric-number bg-gradient-to-r from-sky-400 to-blue-500 bg-clip-text text-transparent font-bold">0.879;</span>
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <!-- 现代化统计数据展示 -->
      <section class="mb-16">
        <div class="text-center mb-10">
          <div class="flex items-center justify-center gap-3 mb-4">
            <div
              class="w-12 h-12 bg-gradient-to-br from-purple-500 to-blue-500 rounded-xl flex items-center justify-center shadow-lg">
              <i class="fas fa-chart-pie text-2xl text-white"></i>
            </div>
            <h2
              class="text-4xl font-bold bg-gradient-to-r from-blue-600 via-purple-600 to-blue-800 bg-clip-text text-transparent">
              Current Statistics
            </h2>
          </div>
          <p class="text-gray-600 text-lg">Real-time data from our comprehensive ADMET database</p>
        </div>

        <!-- 统计卡片网格 (上) -->
        <div class="grid grid-cols-2 md:grid-cols-5 lg:grid-cols-10 gap-6 mb-8 pt-4 pr-4">
          <div v-for="(stat, index) in topStats" :key="stat.name"
            class="group relative bg-gradient-to-br from-white to-blue-50 rounded-2xl shadow-lg border border-blue-100 p-4 hover:shadow-2xl hover:scale-105 transition-all duration-300 cursor-pointer h-24 mt-4 mr-4"
            :style="{ 'animation-delay': `${index * 0.1}s` }" @mouseenter="animateCard($event)"
            @mouseleave="resetCard($event)">

            <!-- 背景装饰 -->
            <div
              class="absolute top-0 right-0 w-16 h-16 bg-gradient-to-br from-blue-400/20 to-purple-400/20 rounded-full -translate-y-8 translate-x-8 overflow-hidden">
            </div>

            <!-- 数值标签 -->
            <div
              class="absolute -top-3 -right-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white text-xs font-bold px-3 py-1.5 rounded-full shadow-lg animate-bounce-subtle z-10">
              {{ stat.value }}
            </div>

            <!-- 卡片内容 -->
            <div class="relative h-full flex flex-col justify-center items-center text-center">
              <i
                class="fas fa-chart-bar text-2xl text-blue-500 mb-2 group-hover:text-purple-600 transition-colors duration-300"></i>
              <span
                class="text-xs font-semibold text-gray-700 group-hover:text-gray-900 transition-colors duration-300">{{
                stat.name }}</span>
            </div>

            <!-- 悬停效果 -->
            <div
              class="absolute inset-0 bg-gradient-to-r from-blue-500/0 to-purple-500/0 group-hover:from-blue-500/10 group-hover:to-purple-500/10 rounded-2xl transition-all duration-300 overflow-hidden">
            </div>
          </div>
        </div>

        <!-- 统计卡片网格 (下) -->
        <div class="grid grid-cols-2 md:grid-cols-5 lg:grid-cols-10 gap-6 pt-4 pr-4">
          <div v-for="(stat, index) in bottomStats" :key="stat.name"
            class="group relative bg-gradient-to-br from-white to-purple-50 rounded-2xl shadow-lg border border-purple-100 p-4 hover:shadow-2xl hover:scale-105 transition-all duration-300 cursor-pointer h-24 mt-4 mr-4"
            :style="{ 'animation-delay': `${(index + 10) * 0.1}s` }" @mouseenter="animateCard($event)"
            @mouseleave="resetCard($event)">

            <!-- 背景装饰 -->
            <div
              class="absolute top-0 right-0 w-16 h-16 bg-gradient-to-br from-purple-400/20 to-pink-400/20 rounded-full -translate-y-8 translate-x-8 overflow-hidden">
            </div>

            <!-- 数值标签 -->
            <div
              class="absolute -top-3 -right-3 bg-gradient-to-r from-purple-500 to-pink-600 text-white text-xs font-bold px-3 py-1.5 rounded-full shadow-lg animate-bounce-subtle z-10">
              {{ stat.value }}
            </div>

            <!-- 卡片内容 -->
            <div class="relative h-full flex flex-col justify-center items-center text-center">
              <i
                class="fas fa-chart-line text-2xl text-purple-500 mb-2 group-hover:text-pink-600 transition-colors duration-300"></i>
              <span
                class="text-xs font-semibold text-gray-700 group-hover:text-gray-900 transition-colors duration-300">{{
                stat.name }}</span>
            </div>

            <!-- 悬停效果 -->
            <div
              class="absolute inset-0 bg-gradient-to-r from-purple-500/0 to-pink-500/0 group-hover:from-purple-500/10 group-hover:to-pink-500/10 rounded-2xl transition-all duration-300 overflow-hidden">
            </div>
          </div>
        </div>
      </section>

      <!-- 现代化功能模块 -->
      <div class="grid lg:grid-cols-2 gap-12">
        <!-- What's ADMETchem -->
        <section class="group">
          <div
            class="relative bg-gradient-to-br from-white via-blue-50 to-indigo-50 rounded-3xl shadow-xl border border-blue-100 overflow-hidden hover:shadow-2xl transition-all duration-500">
            <!-- 背景装饰 -->
            <div
              class="absolute top-0 right-0 w-32 h-32 bg-gradient-to-br from-blue-400/10 to-purple-400/10 rounded-full -translate-y-16 translate-x-16">
            </div>
            <div
              class="absolute bottom-0 left-0 w-24 h-24 bg-gradient-to-tr from-cyan-400/10 to-blue-400/10 rounded-full translate-y-12 -translate-x-12">
            </div>

            <div class="relative p-8">
              <h2 class="text-3xl font-bold mb-6 flex items-center">
                <div class="bg-gradient-to-r from-blue-500 to-purple-600 p-3 rounded-2xl mr-4 shadow-lg">
                  <i class="fas fa-vial text-white text-xl"></i>
                </div>
                <span class="bg-gradient-to-r from-blue-600 to-purple-700 bg-clip-text text-transparent">What's
                  ADMETchem</span>
              </h2>

              <div class="space-y-6">
                <p class="text-gray-700 leading-relaxed text-base">
                  The main reasons for R&D failure are the efficacy and safety deficiencies which are related in part to
                  absorption, distribution, metabolism and excretion (ADME) properties and various toxicities (T) or
                  adverse
                  side effects.
                </p>

                <p class="text-gray-700 leading-relaxed text-base">
                  Here, we developed a platform named
                  <span
                    class="bg-gradient-to-r from-blue-500 to-purple-600 text-white font-bold px-3 py-1 rounded-full shadow-lg">ADMETchem</span>
                  which provides a user-friendly, freely available web interface for systematic ADMET evaluation of
                  chemical compounds based on a comprehensive database consisting of
                  <span
                    class="bg-gradient-to-r from-red-500 to-orange-500 text-white font-bold px-3 py-1 rounded-full shadow-lg animate-pulse">288,967</span>
                  entries.
                </p>

                <p class="text-gray-700 leading-relaxed text-base">
                  Four function modules enable users to perform druglikenness analysis,
                  <span
                    class="bg-gradient-to-r from-green-500 to-emerald-600 text-white font-bold px-2 py-1 rounded-lg shadow-md">31</span>
                  ADMET
                  endpoints prediction, systematic evaluation and similarity searching, which gives an overall
                  understanding of a batch of compounds and can realize the rapid screening process.
                </p>
              </div>
            </div>
          </div>
        </section>

        <!-- Main functionalities -->
        <section class="group">
          <div
            class="relative bg-gradient-to-br from-white via-purple-50 to-pink-50 rounded-3xl shadow-xl border border-purple-100 overflow-hidden hover:shadow-2xl transition-all duration-500">
            <!-- 背景装饰 -->
            <div
              class="absolute top-0 right-0 w-32 h-32 bg-gradient-to-br from-purple-400/10 to-pink-400/10 rounded-full -translate-y-16 translate-x-16">
            </div>
            <div
              class="absolute bottom-0 left-0 w-24 h-24 bg-gradient-to-tr from-indigo-400/10 to-purple-400/10 rounded-full translate-y-12 -translate-x-12">
            </div>

            <div class="relative p-8">
              <h2 class="text-3xl font-bold mb-6 flex items-center">
                <div class="bg-gradient-to-r from-purple-500 to-pink-600 p-3 rounded-2xl mr-4 shadow-lg">
                  <i class="fas fa-cogs text-white text-xl"></i>
                </div>
                <span class="bg-gradient-to-r from-purple-600 to-pink-700 bg-clip-text text-transparent">Main
                  functionalities</span>
              </h2>

              <div class="space-y-4">
                <!-- <a href="#"
                  class="group/item flex items-center justify-between p-5 rounded-2xl bg-gradient-to-r from-white to-blue-50 hover:from-blue-50 hover:to-purple-50 border border-blue-100 hover:border-purple-200 hover:shadow-lg transition-all duration-300 transform hover:scale-[1.02]">
                <div class="flex items-center">
                    <div class="bg-gradient-to-r from-green-400 to-emerald-500 p-2 rounded-xl mr-4 shadow-md">
                      <i class="fas fa-check-circle text-white text-lg"></i>
                    </div>
                    <span class="text-gray-800 font-medium group-hover/item:text-blue-800 transition-colors duration-200">Druglikenness analysis</span>
                </div>
                  <i class="fas fa-arrow-right text-gray-400 group-hover/item:text-blue-600 group-hover/item:translate-x-2 transition-all duration-200"></i>
              </a> -->

                <a href="javascript:void(0);"
                  class="group/item flex items-center justify-between p-5 rounded-2xl bg-gradient-to-r from-white to-purple-50 hover:from-purple-50 hover:to-pink-50 border border-purple-100 hover:border-pink-200 hover:shadow-lg transition-all duration-300 transform hover:scale-[1.02]">
                  <div class="flex items-center">
                    <div class="bg-gradient-to-r from-blue-400 to-cyan-500 p-2 rounded-xl mr-4 shadow-md">
                      <i class="fas fa-check-circle text-white text-lg"></i>
                    </div>
                    <span
                      class="text-gray-800 font-medium group-hover/item:text-purple-800 transition-colors duration-200">ADME/T
                      evaluation</span>
                  </div>
                  <i
                    class="fas fa-arrow-right text-gray-400 group-hover/item:text-purple-600 group-hover/item:translate-x-2 transition-all duration-200"></i>
                </a>

                <a href="javascript:void(0);"
                  class="group/item flex items-center justify-between p-5 rounded-2xl bg-gradient-to-r from-white to-indigo-50 hover:from-indigo-50 hover:to-blue-50 border border-indigo-100 hover:border-blue-200 hover:shadow-lg transition-all duration-300 transform hover:scale-[1.02]">
                  <div class="flex items-center">
                    <div class="bg-gradient-to-r from-purple-400 to-pink-500 p-2 rounded-xl mr-4 shadow-md">
                      <i class="fas fa-check-circle text-white text-lg"></i>
                    </div>
                    <span
                      class="text-gray-800 font-medium group-hover/item:text-indigo-800 transition-colors duration-200">Similarity
                      searching based on ADME/T database</span>
                  </div>
                  <i
                    class="fas fa-arrow-right text-gray-400 group-hover/item:text-indigo-600 group-hover/item:translate-x-2 transition-all duration-200"></i>
                </a>

                <router-link to="/optimization"
                  class="group/item flex items-center justify-between p-5 rounded-2xl bg-gradient-to-r from-white to-cyan-50 hover:from-cyan-50 hover:to-teal-50 border border-cyan-100 hover:border-teal-200 hover:shadow-lg transition-all duration-300 transform hover:scale-[1.02]">
                  <div class="flex items-center">
                    <div class="bg-gradient-to-r from-indigo-400 to-purple-500 p-2 rounded-xl mr-4 shadow-md">
                      <i class="fas fa-check-circle text-white text-lg"></i>
                    </div>
                    <span
                      class="text-gray-800 font-medium group-hover/item:text-cyan-800 transition-colors duration-200">Molecular
                      Optimization</span>
                  </div>
                  <i
                    class="fas fa-arrow-right text-gray-400 group-hover/item:text-cyan-600 group-hover/item:translate-x-2 transition-all duration-200"></i>
                </router-link>

                <div class="py-8">

                </div>
              </div>
            </div>
          </div>
        </section>
      </div>

      <!-- 装饰性分隔线 -->
      <div class="my-16 flex items-center justify-center">
        <div class="flex-1 h-px bg-gradient-to-r from-transparent via-gray-300 to-transparent"></div>
        <div class="mx-4 p-3 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full shadow-lg">
          <i class="fas fa-dna text-white text-xl"></i>
        </div>
        <div class="flex-1 h-px bg-gradient-to-r from-transparent via-gray-300 to-transparent"></div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import flowchart1 from '@/assets/index-1.jpg'
import flowchart2 from '@/assets/index-2.gif'

// 统计数据
const topStats = ref([
  { name: 'LogS', value: '8605' },
  { name: 'LogD7.4', value: '4419' },
  { name: 'Caco-2', value: '1182' },
  { name: 'Pgp-Inhibitor', value: '2297' },
  { name: 'Pgp-Substrate', value: '1252' },
  { name: 'HIA', value: '1997' },
  { name: 'F(20%)', value: '2555' },
  { name: 'F(30%)', value: '2555' },
  { name: 'VD', value: '1864' },
  { name: 'CYP Inhibitor/Substrate', value: '64084' }
])

const bottomStats = ref([
  { name: 'BBB', value: '5448' },
  { name: 'Clearance', value: '7641' },
  { name: 'T1/2', value: '2855' },
  { name: 'PPB', value: '6690' },
  { name: 'hERG', value: '2171' },
  { name: 'SkinSen', value: '663' },
  { name: 'Ames', value: '7619' },
  { name: 'DILI', value: '475' },
  { name: 'FDMDD', value: '873' },
  { name: 'Acute toxicity', value: '7413' }
])

// 卡片动画函数
const animateCard = (event) => {
  const card = event.currentTarget
  card.style.transform = 'translateY(-8px) scale(1.05)'
  card.style.transition = 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)'
}

const resetCard = (event) => {
  const card = event.currentTarget
  card.style.transform = 'translateY(0) scale(1)'
  card.style.transition = 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)'
}
</script>

<style scoped>
/* 现代化动画效果 */
@keyframes shimmer {
  0% {
    transform: translateX(-100%);
  }

  100% {
    transform: translateX(100%);
  }
}

@keyframes gradient {

  0%,
  100% {
    background-position: 0% 50%;
  }

  50% {
    background-position: 100% 50%;
  }
}

@keyframes bounce-subtle {

  0%,
  100% {
    transform: translateY(0);
  }

  50% {
    transform: translateY(-4px);
  }
}

@keyframes marquee {
  0% {
    transform: translateX(0%);
    opacity: 1;
  }

  100% {
    transform: translateX(-100%);
    opacity: 1;
  }
}

/* 字幕淡入动画 */
@keyframes fadeIn {
  0% {
    opacity: 0;
  }

  100% {
    opacity: 1;
  }
}

.animate-shimmer {
  animation: shimmer 3s infinite linear;
}

.animate-gradient {
  background-size: 200% 200%;
  animation: gradient 6s ease infinite;
}

.animate-bounce-subtle {
  animation: bounce-subtle 2s infinite;
}

/* 滚动字幕样式 */
.marquee-container {
  overflow: hidden;
  white-space: nowrap;
  position: relative;
  animation: fadeIn 0.8s ease-out;
}

.marquee-content {
  display: inline-block;
  animation: marquee 30s linear infinite;
  animation-delay: 0s;
  width: max-content;
  position: relative;
  /* 立即显示内容 */
  transform: translateX(0%);
}

.marquee-content:hover {
  animation-play-state: paused;
}

.metric-number {
  font-size: 1.1em;
  font-weight: 800;
  text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

/* 卡片进入动画 */
@keyframes slideInUp {
  from {
    opacity: 0;
    transform: translateY(30px);
  }

  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.grid>div {
  animation: slideInUp 0.6s ease-out forwards;
  opacity: 0;
}

/* 响应式优化 */
@media (max-width: 768px) {
  .marquee-content {
    animation-duration: 20s;
  }

  .metric-number {
    font-size: 1em;
  }
}

/* 自定义滚动条 */
::-webkit-scrollbar {
  width: 8px;
}

::-webkit-scrollbar-track {
  background: #f1f5f9;
  border-radius: 4px;
}

::-webkit-scrollbar-thumb {
  background: linear-gradient(to bottom, #3b82f6, #8b5cf6);
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: linear-gradient(to bottom, #2563eb, #7c3aed);
}

/* 焦点样式优化 */
a:focus-visible {
  outline: 2px solid #3b82f6;
  outline-offset: 2px;
  border-radius: 8px;
}
</style>