<template>
  <div class="min-h-screen bg-gray-50">


    <!-- 主要内容区域 -->
    <div class="max-w-7xl mx-auto px-4">
            <!-- 头部说明 -->
            <div class="mt-8 bg-blue-50 border border-blue-200 rounded-lg p-6">
        <div class="flex items-start space-x-3">
          <i class="fas fa-info-circle text-blue-500 mt-1"></i>
          <div class="text-sm text-gray-700 leading-relaxed">
            <p class="font-medium text-gray-800 mb-2">
              ★ A smart ADME/T property searching engine with the abilities to search by molecular structures or pharmacokinetic properties, and by fingerprint similarity.
            </p>
            <p class="mb-2">
              <strong>Caveat:</strong> ADMETchem is provided free-of-charge in hope that it will be useful, but you must use it at your own risk.
            </p>
            <p>
              If you would like to use ADMETchem securely, please <a href="#" class="text-blue-600 hover:underline">contact us</a>!
            </p>
          </div>
        </div>
      </div>

      <!-- 搜索区域 -->
      <div class="grid grid-cols-1 gap-8 py-8 mt-8">
        
        <!-- Property Search - 上侧 -->
        <div class="bg-white rounded-lg  border border-blue-200">
          <div class="p-6">
            <h2 class="text-2xl font-bold text-gray-800 mb-6 text-center">Property Search</h2>
            
            <!-- 选项卡切换 -->
            <div class="flex mb-6 border-b border-gray-200">
              <button 
                @click="propertySearchType = 'accurate'"
                :class="[
                  'px-4 py-2 font-medium text-sm border-b-2 transition-colors duration-200',
                  propertySearchType === 'accurate' 
                    ? 'text-blue-600 border-blue-600' 
                    : 'text-gray-500 border-transparent hover:text-gray-700'
                ]"
              >
                <i class="fas fa-crosshairs mr-2"></i>Accurate Search
              </button>
              <button 
                @click="propertySearchType = 'range'"
                :class="[
                  'px-4 py-2 font-medium text-sm border-b-2 transition-colors duration-200 ml-4',
                  propertySearchType === 'range' 
                    ? 'text-blue-600 border-blue-600' 
                    : 'text-gray-500 border-transparent hover:text-gray-700'
                ]"
              >
                <i class="fas fa-sliders-h mr-2"></i>Range Search
              </button>
            </div>

            <!-- Accurate Search 表单 -->
            <div v-if="propertySearchType === 'accurate'" class="space-y-4">
              <!-- SMILES -->
              <div>
                <label class="block text-sm font-medium text-gray-700 mb-2">SMILES:</label>
                <div class="flex space-x-2">
                  <input 
                    v-model="propertyForm.smiles"
                    type="text" 
                    placeholder="Example: O=C([O-])C(C)c1ccc(cc1)C(C)C"
                    class="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                  <button 
                    @click="selectExample('smiles')"
                    class="px-3 py-2 text-sm bg-gray-100 text-gray-700 border border-gray-300 rounded-md hover:bg-gray-200 transition-colors duration-200"
                  >
                    Select
                  </button>
                </div>
                <p class="text-xs text-gray-500 mt-1">e.g., <span class="text-blue-600 cursor-pointer hover:underline" @click="setExample('smiles')">ibuprofen</span></p>
              </div>

              <!-- CAS RN -->
              <div>
                <label class="block text-sm font-medium text-gray-700 mb-2">CAS RN:</label>
                <div class="flex space-x-2">
                  <input 
                    v-model="propertyForm.casRn"
                    type="text" 
                    placeholder="Example: 15687-27-1"
                    class="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                  <button 
                    @click="selectExample('casRn')"
                    class="px-3 py-2 text-sm bg-gray-100 text-gray-700 border border-gray-300 rounded-md hover:bg-gray-200 transition-colors duration-200"
                  >
                    Select
                  </button>
                </div>
                <p class="text-xs text-gray-500 mt-1">e.g., <span class="text-blue-600 cursor-pointer hover:underline" @click="setExample('casRn')">ibuprofen</span></p>
              </div>

              <!-- IUPAC Name -->
              <div>
                <label class="block text-sm font-medium text-gray-700 mb-2">IUPAC Name:</label>
                <div class="flex space-x-2">
                  <input 
                    v-model="propertyForm.iupacName"
                    type="text" 
                    placeholder="Example: 2-[4-(2-methylpropyl)phenyl]propanoate"
                    class="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                  <button 
                    @click="selectExample('iupacName')"
                    class="px-3 py-2 text-sm bg-gray-100 text-gray-700 border border-gray-300 rounded-md hover:bg-gray-200 transition-colors duration-200"
                  >
                    Select
                  </button>
                </div>
                <p class="text-xs text-gray-500 mt-1">e.g., <span class="text-blue-600 cursor-pointer hover:underline" @click="setExample('iupacName')">ibuprofen</span></p>
              </div>
            </div>

            <!-- Range Search 表单 -->
            <div v-if="propertySearchType === 'range'" class="space-y-4">
              <!-- Molecular Weight -->
              <div>
                <label class="block text-sm font-medium text-gray-700 mb-2">Molecular Weight</label>
                <div class="flex items-center space-x-2">
                  <input 
                    v-model="rangeForm.molecularWeight.min"
                    type="number" 
                    placeholder="1"
                    class="w-20 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                  <span class="text-gray-500">-</span>
                  <input 
                    v-model="rangeForm.molecularWeight.max"
                    type="number" 
                    placeholder="1500"
                    class="w-20 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                  <span class="text-sm text-gray-500">Limit: 1:1500</span>
                  <button class="px-3 py-2 text-sm bg-gray-100 text-gray-700 border border-gray-300 rounded-md hover:bg-gray-200 transition-colors duration-200">
                    Select
                  </button>
                </div>
              </div>

              <!-- ALOGP -->
              <div>
                <label class="block text-sm font-medium text-gray-700 mb-2">ALOGP</label>
                <div class="flex items-center space-x-2">
                  <input 
                    v-model="rangeForm.alogp.min"
                    type="number" 
                    placeholder="-15"
                    step="0.1"
                    class="w-20 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                  <span class="text-gray-500">-</span>
                  <input 
                    v-model="rangeForm.alogp.max"
                    type="number" 
                    placeholder="15"
                    step="0.1"
                    class="w-20 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                  <span class="text-sm text-gray-500">Limit: -15:15</span>
                  <button class="px-3 py-2 text-sm bg-gray-100 text-gray-700 border border-gray-300 rounded-md hover:bg-gray-200 transition-colors duration-200">
                    Select
                  </button>
                </div>
              </div>

              <!-- HBA -->
              <div>
                <label class="block text-sm font-medium text-gray-700 mb-2">HBA</label>
                <div class="flex items-center space-x-2">
                  <input 
                    v-model="rangeForm.hba.min"
                    type="number" 
                    placeholder="0"
                    class="w-20 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                  <span class="text-gray-500">-</span>
                  <input 
                    v-model="rangeForm.hba.max"
                    type="number" 
                    placeholder="20"
                    class="w-20 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                  <span class="text-sm text-gray-500">Limit: 0:20</span>
                  <button class="px-3 py-2 text-sm bg-gray-100 text-gray-700 border border-gray-300 rounded-md hover:bg-gray-200 transition-colors duration-200">
                    Select
                  </button>
                </div>
              </div>

              <!-- HBD -->
              <div>
                <label class="block text-sm font-medium text-gray-700 mb-2">HBD</label>
                <div class="flex items-center space-x-2">
                  <input 
                    v-model="rangeForm.hbd.min"
                    type="number" 
                    placeholder="0"
                    class="w-20 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                  <span class="text-gray-500">-</span>
                  <input 
                    v-model="rangeForm.hbd.max"
                    type="number" 
                    placeholder="20"
                    class="w-20 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                  <span class="text-sm text-gray-500">Limit: 0:20</span>
                  <button class="px-3 py-2 text-sm bg-gray-100 text-gray-700 border border-gray-300 rounded-md hover:bg-gray-200 transition-colors duration-200">
                    Select
                  </button>
                </div>
              </div>
            </div>

            <!-- 搜索和重置按钮 -->
            <div class="flex space-x-4 mt-6">
              <button 
                @click="performPropertySearch"
                class="flex-1 bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600 transition-colors duration-200 flex items-center justify-center"
                :disabled="isSearching"
              >
                <i class="fas fa-search mr-2"></i>
                {{ isSearching ? 'Searching...' : 'Search' }}
              </button>
              <button 
                @click="resetPropertyForm"
                class="px-6 py-2 bg-gray-100 text-gray-700 border border-gray-300 rounded-md hover:bg-gray-200 transition-colors duration-200"
              >
                Reset
              </button>
            </div>
          </div>
        </div>

        <!-- Similarity Search - 下侧 -->
        <div class="bg-white rounded-lg border border-blue-200">
          <div class="p-6">
            <h2 class="text-2xl font-bold text-gray-800 mb-6 text-center">Similarity Search</h2>
            
            <div class="space-y-4">
              <!-- SMILES -->
              <div>
                <label class="block text-sm font-medium text-gray-700 mb-2">SMILES:</label>
                <input 
                  v-model="similarityForm.smiles"
                  type="text" 
                  placeholder="Example: CC(C)CC1=CC=C(C=C1)C(C)C(O)=O"
                  class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                <p class="text-xs text-gray-500 mt-1">e.g., <span class="text-blue-600 cursor-pointer hover:underline" @click="setSimilarityExample()">ibuprofen</span></p>
              </div>

              <!-- Select fingerprinter -->
              <div>
                <label class="block text-sm font-medium text-gray-700 mb-2">Select fingerprinter</label>
                <select 
                  v-model="similarityForm.fingerprinter"
                  class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white"
                >
                  <option value="maccs">MACCS fingerprint</option>
                  <option value="daylight">Daylight fingerprint</option>
                  <option value="atom-pairs">Atom pairs fingerprint</option>
                  <option value="topological-torsion">Topological Torsion Fingerprint</option>
                  <option value="morgan">Morgan Fingerprint (radius=2)</option>
                </select>
              </div>

              <!-- Select similarity metric -->
              <div>
                <label class="block text-sm font-medium text-gray-700 mb-2">Select similarity metric</label>
                <div class="flex space-x-4">
                  <label class="flex items-center">
                    <input 
                      v-model="similarityForm.metric"
                      type="radio" 
                      value="tanimoto"
                      class="mr-2 text-blue-600 focus:ring-blue-500"
                    >
                    <span class="text-sm text-gray-700">Tanimoto</span>
                  </label>
                  <label class="flex items-center">
                    <input 
                      v-model="similarityForm.metric"
                      type="radio" 
                      value="dice"
                      class="mr-2 text-blue-600 focus:ring-blue-500"
                    >
                    <span class="text-sm text-gray-700">Dice</span>
                  </label>
                </div>
              </div>

              <!-- Similarity threshold -->
              <div>
                <label class="block text-sm font-medium text-gray-700 mb-2">Similarity:</label>
                <input 
                  v-model="similarityForm.similarity"
                  type="number" 
                  step="0.1"
                  min="0.2"
                  max="1.0"
                  placeholder="Example: 0.7"
                  class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                <p class="text-xs text-gray-500 mt-1">e.g., 0.7; Limit: 0.2-1.0</p>
              </div>
            </div>

            <!-- 搜索和重置按钮 -->
            <div class="flex space-x-4 mt-6">
              <button 
                @click="performSimilaritySearch"
                class="flex-1 bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600 transition-colors duration-200 flex items-center justify-center"
                :disabled="isSearching"
              >
                <i class="fas fa-search mr-2"></i>
                {{ isSearching ? 'Searching...' : 'Search' }}
              </button>
              <button 
                @click="resetSimilarityForm"
                class="px-6 py-2 bg-gray-100 text-gray-700 border border-gray-300 rounded-md hover:bg-gray-200 transition-colors duration-200"
              >
                Reset
              </button>
            </div>
          </div>
        </div>
      </div>


    </div>
  </div>
</template>

<script setup>
import { ref, reactive } from 'vue'
import Breadcrumb from '@/components/Breadcrumb.vue'

// 搜索类型
const propertySearchType = ref('accurate')
const isSearching = ref(false)

// Property Search 表单数据
const propertyForm = reactive({
  smiles: '',
  casRn: '',
  iupacName: ''
})

// Range Search 表单数据
const rangeForm = reactive({
  molecularWeight: { min: '', max: '' },
  alogp: { min: '', max: '' },
  hba: { min: '', max: '' },
  hbd: { min: '', max: '' }
})

// Similarity Search 表单数据
const similarityForm = reactive({
  smiles: '',
  fingerprinter: 'maccs',
  metric: 'tanimoto',
  similarity: ''
})

// 示例数据
const examples = {
  smiles: 'CC(C)CC1=CC=C(C=C1)C(C)C(O)=O',
  casRn: '15687-27-1',
  iupacName: '2-[4-(2-methylpropyl)phenyl]propanoate'
}

// 设置示例数据
const setExample = (field) => {
  propertyForm[field] = examples[field]
}

const setSimilarityExample = () => {
  similarityForm.smiles = examples.smiles
}

const selectExample = (field) => {
  // 这里可以添加更复杂的选择逻辑
  setExample(field)
}

// 执行属性搜索
const performPropertySearch = async () => {
  isSearching.value = true
  try {
    // 这里添加实际的搜索逻辑
    console.log('Property Search:', propertySearchType.value, propertyForm, rangeForm)
    
    // 模拟API调用
    await new Promise(resolve => setTimeout(resolve, 1000))
    
    // 显示搜索结果（可以使用 toast 或者跳转到结果页面）
    alert('Property search completed! (Demo)')
  } catch (error) {
    console.error('Property search error:', error)
    alert('Search failed. Please try again.')
  } finally {
    isSearching.value = false
  }
}

// 执行相似性搜索
const performSimilaritySearch = async () => {
  isSearching.value = true
  try {
    // 这里添加实际的搜索逻辑
    console.log('Similarity Search:', similarityForm)
    
    // 模拟API调用
    await new Promise(resolve => setTimeout(resolve, 1000))
    
    // 显示搜索结果
    alert('Similarity search completed! (Demo)')
  } catch (error) {
    console.error('Similarity search error:', error)
    alert('Search failed. Please try again.')
  } finally {
    isSearching.value = false
  }
}

// 重置表单
const resetPropertyForm = () => {
  Object.assign(propertyForm, {
    smiles: '',
    casRn: '',
    iupacName: ''
  })
  
  Object.assign(rangeForm, {
    molecularWeight: { min: '', max: '' },
    alogp: { min: '', max: '' },
    hba: { min: '', max: '' },
    hbd: { min: '', max: '' }
  })
}

const resetSimilarityForm = () => {
  Object.assign(similarityForm, {
    smiles: '',
    fingerprinter: 'maccs',
    metric: 'tanimoto',
    similarity: ''
  })
}
</script>

<style scoped>
/* 自定义样式 */
input:focus, select:focus {
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

/* 响应式调整 */
@media (max-width: 1024px) {
  .grid-cols-1.lg\:grid-cols-2 {
    gap: 1.5rem;
  }
}
</style> 