<template>
  <div class="bg-gray-50 min-h-screen">
    <main class="max-w-4xl mx-auto px-4 py-12">
      <!-- 主标题 -->
      <div class="text-center mb-12">
        <h1 class="text-4xl font-bold text-gray-800 mb-4">Input SMILES</h1>
      </div>

      <!-- SMILES 输入区域 -->
      <div class="bg-white rounded-lg border border-slate-200 p-8 mb-8">
        <div class="flex items-center space-x-4 mb-6">
          <input
            type="text"
            v-model="smilesInput"
            placeholder="Example: CC1=CN=C(C=C1OC)C(=O)C2=NC3=C(N2)C=C(C=C3)OC"
            class="flex-1 px-4 py-3 text-lg border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
          <button
            @click="handleSearch"
            class="bg-green-600 text-white px-8 py-3 rounded-lg font-medium hover:bg-green-700 transition duration-200 flex items-center"
          >
            <i class="fas fa-search mr-2"></i>
            Submit
          </button>
        </div>

        <!-- 示例按钮 -->
        <div class="flex justify-center space-x-4">
          <button
            @click="loadExample('omeprazole')"
            class="bg-gray-200 text-gray-700 px-6 py-2 rounded-lg hover:bg-gray-300 transition duration-200"
          >
            Example: Omeprazole
          </button>
          <button
            @click="loadExample('abacavir')"
            class="bg-gray-200 text-gray-700 px-6 py-2 rounded-lg hover:bg-gray-300 transition duration-200"
          >
            Example: Abacavir
          </button>
        </div>
      </div>

      <!-- 分隔线 -->
      <div class="border border-slate-200 my-8"></div>

      <!-- 说明文字 -->
      <div class="bg-blue-50 border border-blue-200 rounded-lg p-6 text-center">
        <div class="text-lg font-semibold text-gray-800 mb-3">
          <i class="fas fa-star text-yellow-500 mr-2"></i>
          S indices represent the similarity between each molecule and the training dataset by using Tanimoto method and MACCS fingerprints.
        </div>

        <div class="space-y-2 text-sm text-gray-600">
          <div class="flex items-start justify-center">
            <i class="fas fa-exclamation-triangle text-orange-500 mr-2 mt-0.5"></i>
            <div>
              <strong>Caveat:</strong> ADMETchem is provided free-of-charge in hope that it will be useful, but you must use it at your own risk.
            </div>
          </div>

          <div class="flex items-center justify-center mt-3">
            <i class="fas fa-envelope text-blue-500 mr-2"></i>
            <span>If you would like to use ADMETchem securely, please
              <a href="#" class="text-blue-600 hover:text-blue-800 underline font-medium">contact us</a>!
            </span>
          </div>
        </div>
      </div>

      <!-- 功能特性说明 -->
      <div class="mt-12 grid md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div class="bg-white rounded-lg border border-slate-200 p-6 text-center">
          <div class="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <i class="fas fa-flask text-blue-600 text-xl"></i>
          </div>
          <h3 class="font-semibold text-gray-800 mb-2">Physicochemical</h3>
          <p class="text-sm text-gray-600">LogS, LogD, LogP properties prediction</p>
        </div>

        <div class="bg-white rounded-lg border border-slate-200 p-6 text-center">
          <div class="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <i class="fas fa-arrow-down text-green-600 text-xl"></i>
          </div>
          <h3 class="font-semibold text-gray-800 mb-2">Absorption</h3>
          <p class="text-sm text-gray-600">Caco-2, Pgp, HIA, Bioavailability</p>
        </div>

        <div class="bg-white rounded-lg border border-slate-200 p-6 text-center">
          <div class="w-12 h-12 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <i class="fas fa-share-alt text-purple-600 text-xl"></i>
          </div>
          <h3 class="font-semibold text-gray-800 mb-2">Distribution</h3>
          <p class="text-sm text-gray-600">PPB, BBB, Volume Distribution</p>
        </div>

        <div class="bg-white rounded-lg border border-slate-200 p-6 text-center">
          <div class="w-12 h-12 bg-orange-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <i class="fas fa-cogs text-orange-600 text-xl"></i>
          </div>
          <h3 class="font-semibold text-gray-800 mb-2">Metabolism</h3>
          <p class="text-sm text-gray-600">CYP450 enzymes interaction</p>
        </div>
      </div>

      <!-- 底部额外信息 -->
      <div class="mt-12 text-center">
        <div class="bg-gradient-to-r from-green-50 to-blue-50 rounded-lg p-6 border border-green-200">
          <h3 class="text-lg font-semibold text-gray-800 mb-3 flex items-center justify-center">
            <i class="fas fa-chart-line text-green-600 mr-2"></i>
            Application Domain Analysis
          </h3>
          <p class="text-gray-600 max-w-2xl mx-auto">
            Evaluate the applicability domain of your compounds using advanced similarity analysis and machine learning techniques to ensure reliable predictions.
          </p>
        </div>
      </div>
    </main>
  </div>
</template>

<script setup>
import { ref } from 'vue'

// SMILES 输入
const smilesInput = ref('')

// 示例化合物的 SMILES
const exampleSmiles = {
  omeprazole: 'CC1=CN=C(C=C1OC)C(=O)C2=NC3=C(N2)C=C(C=C3)OC',
  abacavir: 'NC1=NC(NC2CC2)=C3N=CNC3=N1'
}

// 加载示例
const loadExample = (type) => {
  if (exampleSmiles[type]) {
    smilesInput.value = exampleSmiles[type]
  }
}

// 搜索处理
const handleSearch = () => {
  if (!smilesInput.value.trim()) {
    alert('请输入有效的 SMILES 结构')
    return
  }
  
  // 这里将来会添加搜索逻辑
  console.log('搜索SMILES:', smilesInput.value)
  alert('应用领域分析功能正在开发中...')
}
</script>

<style scoped>
/* 自定义样式 */
.transition-all {
  transition: all 0.3s ease;
}

/* 按钮悬停效果 */
button:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}
</style> 