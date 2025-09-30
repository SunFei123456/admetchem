<template>
  <div class="bg-gray-50 min-h-screen">
    <main class="max-w-7xl mx-auto px-4 py-6">
      <!-- 成功提示 -->
      <div class="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
        <div class="flex items-center">
          <div class="bg-green-500 text-white px-3 py-1 rounded text-sm font-medium">
            Success!
          </div>
        </div>
      </div>

      <!-- 结果展示 -->
      <div class="bg-white rounded-lg border border-slate-200">
        <!-- 标题栏 -->
        <div class="bg-green-500 text-white px-4 py-3 flex items-center">
          <i class="fas fa-check-circle mr-2"></i>
          <span class="font-medium">{{ selectedRuleName }}</span>
        </div>

        <!-- 分页控制 -->
        <div class="p-4 border-b border-gray-200 flex justify-between items-center">
          <div class="flex items-center space-x-2">
            <select v-model="pageSize" @change="currentPage = 1" class="border border-gray-300 rounded px-2 py-1 text-sm">
              <option value="10">10</option>
              <option value="20">20</option>
              <option value="50">50</option>
              <option value="100">100</option>
            </select>
            <span class="text-sm text-gray-600">records per page</span>
          </div>
          <div class="flex items-center space-x-4" v-if="dataSource === 'file'">
            <div class="flex items-center space-x-2">
              <span class="text-sm text-gray-600">Search:</span>
              <input v-model="searchQuery" @input="currentPage = 1" type="text" class="border border-gray-300 rounded px-2 py-1 text-sm" placeholder="Search SMILES...">
              <button v-if="searchQuery.trim()" @click="clearSearch" class="text-blue-600 hover:text-blue-800 text-sm">
                <i class="fas fa-times"></i>
              </button>
            </div>
            <div class="flex items-center space-x-2" v-if="sortField">
              <span class="text-sm text-gray-600">Sorted by: {{ tableHeaders[sortField] || (sortField === 'smiles' ? 'Molecule' : sortField === 'matches' ? 'Matches' : sortField) }}</span>
              <button @click="clearSort" class="text-blue-600 hover:text-blue-800 text-sm">
                <i class="fas fa-times"></i> Clear
              </button>
            </div>
          </div>
        </div>

        <!-- 结果表格 -->
        <div class="overflow-x-auto">
          <table class="w-full table-fixed">
            <thead class="bg-gray-50">
              <tr>
                <th class="w-80 px-4 py-3 text-left text-sm font-medium text-gray-700 border-b">
                  <div class="flex items-center cursor-pointer hover:text-blue-600" 
                       @click="dataSource === 'file' && sortBy('smiles')">
                    Molecule
                    <i :class="dataSource === 'file' ? getSortIcon('smiles') : 'fas fa-sort text-gray-400'" class="ml-1"></i>
                  </div>
                </th>
                <th v-for="(header, key) in tableHeaders" :key="key" 
                    class="w-32 px-4 py-3 text-left text-sm font-medium text-gray-700 border-b">
                  <div class="flex items-center cursor-pointer hover:text-blue-600" 
                       @click="dataSource === 'file' && sortBy(key)">
                    {{ header }}
                    <i :class="dataSource === 'file' ? getSortIcon(key) : 'fas fa-sort text-gray-400'" class="ml-1"></i>
                  </div>
                </th>
                <th class="w-24 px-4 py-3 text-left text-sm font-medium text-gray-700 border-b">
                  <div class="flex items-center cursor-pointer hover:text-blue-600" 
                       @click="dataSource === 'file' && sortBy('matches')">
                    Matches
                    <i :class="dataSource === 'file' ? getSortIcon('matches') : 'fas fa-sort text-gray-400'" class="ml-1"></i>
                  </div>
                </th>
              </tr>
            </thead>
            <tbody>
              <!-- 单个分子结果（SMILES输入） -->
              <tr v-if="dataSource === 'smiles'" class="border-b border-gray-200 hover:bg-gray-50">
                <td class="w-80 px-4 py-3">
                  <div class="flex items-center">
                    <button class="text-blue-600 hover:text-blue-800 mr-2">
                      <i class="fas fa-plus-square"></i>
                    </button>
                    <span class="font-mono text-sm truncate">{{ resultData?.smiles || 'N/A' }}</span>
                  </div>
                </td>
                <td v-for="(value, key) in currentMetrics" :key="key" class="w-32 px-4 py-3 text-sm">
                  {{ typeof value === 'number' ? (Math.floor(value * 100) / 100).toString() : value }}
                </td>
                <td class="w-24 px-4 py-3">
                  <span class="bg-blue-500 text-white px-2 py-1 rounded text-sm font-medium">
                    {{ currentMatchPercentage }}%
                  </span>
                </td>
              </tr>
              
              <!-- 多个分子结果（SDF文件上传） -->
              <tr v-for="(item, index) in paginatedItems" :key="index" class="border-b border-gray-200 hover:bg-gray-50">
                <td class="w-80 px-4 py-3">
                  <div class="flex items-center">
                    <button class="text-blue-600 hover:text-blue-800 mr-2">
                      <i class="fas fa-plus-square"></i>
                    </button>
                    <span class="font-mono text-sm truncate">{{ item.smiles }}</span>
                  </div>
                </td>
                <td v-for="(value, key) in item.metrics[selectedRule]" :key="key" class="w-32 px-4 py-3 text-sm">
                  {{ typeof value === 'number' ? (Math.floor(value * 100) / 100).toString() : value }}
                </td>
                <td class="w-24 px-4 py-3">
                  <span class="bg-blue-500 text-white px-2 py-1 rounded text-sm font-medium">
                    {{ (item.matches[selectedRule] * 100).toFixed(2) }}%
                  </span>
                </td>
              </tr>
              
              <!-- 无数据提示 -->
              <tr v-if="(dataSource === 'file' && paginatedItems.length === 0) || (dataSource === 'smiles' && !resultData)">
                <td :colspan="Object.keys(tableHeaders).length + 2" class="px-4 py-8 text-center text-gray-500">
                  暂无数据
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <!-- 分页信息 -->
        <div class="p-4 border-t border-gray-200 flex justify-between items-center">
          <div class="text-sm text-gray-600">
            <span v-if="dataSource === 'smiles'">
              Showing 1 to 1 of 1 entries
            </span>
            <span v-else>
              Showing {{ showingStart }} to {{ showingEnd }} of {{ totalItems }} entries
              <span v-if="searchQuery.trim()">(filtered from {{ resultData?.items?.length || 0 }} total entries)</span>
            </span>
          </div>
          <div class="flex items-center space-x-2" v-if="dataSource === 'file' && totalPages > 1">
            <button @click="currentPage = Math.max(1, currentPage - 1)" 
                    :disabled="currentPage === 1"
                    :class="currentPage === 1 ? 'text-gray-500 cursor-not-allowed' : 'text-blue-600 hover:text-blue-800'"
                    class="px-3 py-1 border border-gray-300 rounded text-sm">
              ← Previous
            </button>
            
            <!-- 页码显示 -->
            <template v-for="page in Math.min(totalPages, 5)" :key="page">
              <button v-if="page <= totalPages" 
                      @click="currentPage = page"
                      :class="currentPage === page ? 'bg-blue-500 text-white' : 'text-blue-600 hover:text-blue-800'"
                      class="px-3 py-1 border border-gray-300 rounded text-sm">
                {{ page }}
              </button>
            </template>
            
            <span v-if="totalPages > 5" class="px-2 text-gray-500">...</span>
            
            <button @click="currentPage = Math.min(totalPages, currentPage + 1)" 
                    :disabled="currentPage === totalPages"
                    :class="currentPage === totalPages ? 'text-gray-500 cursor-not-allowed' : 'text-blue-600 hover:text-blue-800'"
                    class="px-3 py-1 border border-gray-300 rounded text-sm">
              Next →
            </button>
          </div>
          <div v-else-if="dataSource === 'smiles'" class="flex items-center space-x-2">
            <button class="px-3 py-1 border border-gray-300 rounded text-sm text-gray-500 cursor-not-allowed">
              ← Previous
            </button>
            <span class="px-3 py-1 bg-blue-500 text-white rounded text-sm">1</span>
            <button class="px-3 py-1 border border-gray-300 rounded text-sm text-gray-500 cursor-not-allowed">
              Next →
            </button>
          </div>
        </div>
      </div>

      <!-- 操作按钮 -->
      <div class="mt-6 text-center space-x-4">
        <button @click="exportToCSV" 
                class="bg-green-600 text-white px-6 py-2 rounded hover:bg-green-700 transition duration-200">
          <i class="fas fa-download mr-2"></i>
          导出CSV
        </button>
        <button @click="goBack" 
                class="bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-700 transition duration-200">
          <i class="fas fa-arrow-left mr-2"></i>
          返回评估页面
        </button>
      </div>
    </main>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import Breadcrumb from '@/components/Breadcrumb.vue'
import * as XLSX from 'xlsx'
import { getResultData, removeResultData, cleanupExpiredData } from '../utils/storage.js'

const route = useRoute()
const router = useRouter()



// 结果数据
const resultData = ref(null)
const selectedRule = ref('')
const dataSource = ref('smiles') // 'smiles' 或 'file'
const currentPage = ref(1)
const pageSize = ref(10)
const searchQuery = ref('')

// 排序状态
const sortField = ref('')
const sortDirection = ref('asc') // 'asc' 或 'desc'

// 规则名称映射
const ruleNameMapping = {
  'Lipinski': "Lipinski's rules",
  'Ghose': "Ghose's rules",
  'Oprea': "Oprea's rules",
  'Veber': "Veber's rules",
  'Varma': "Varma's rules"
}

// 表格头部映射
const tableHeaderMapping = {
  'Lipinski': {
    'mw': 'Molecular weight',
    'logp': 'LogP',
    'hbd': 'H-bond donors',
    'hba': 'H-bond acceptors'
  },
  'Ghose': {
    'mw': 'Molecular weight',
    'logp': 'LogP',
    'mr': 'Molar refractivity',
    'atom_count': 'Total number of atoms'
  },
  'Oprea': {
    'rot_bonds': 'Rotatable bonds',
    'rigid_bonds': 'Rigid bonds',
    'ring_count': 'Ring count'
  },
  'Veber': {
    'rot_bonds': 'Rotatable bonds',
    'tpsa': 'TPSA',
    'hbd': 'H-bond donors',
    'hba': 'H-bond acceptors'
  },
  'Varma': {
    'molecular_weight': 'Molecular weight',
    'tpsa': 'TPSA',
    'logd': 'LogD',
    'h_bond_donor': 'H-bond donors',
    'h_bond_acceptor': 'H-bond acceptors',
    'rotatable_bonds': 'Rotatable bonds'
  }
}

// 计算属性
const selectedRuleName = computed(() => {
  return ruleNameMapping[selectedRule.value] || selectedRule.value
})

const tableHeaders = computed(() => {
  return tableHeaderMapping[selectedRule.value] || {}
})

// 处理单个分子数据（SMILES输入）
const currentMetrics = computed(() => {
  if (dataSource.value === 'smiles') {
    if (!resultData.value?.metrics || !selectedRule.value) return {}
    return resultData.value.metrics[selectedRule.value] || {}
  }
  return {}
})

const currentMatchPercentage = computed(() => {
  if (dataSource.value === 'smiles') {
    if (!resultData.value?.matches || !selectedRule.value) return '0.00'
    const match = resultData.value.matches[selectedRule.value] || 0
    return (match * 100).toFixed(2)
  }
  return '0.00'
})

// 排序函数
const sortBy = (field) => {
  if (sortField.value === field) {
    // 如果点击的是同一个字段，切换排序方向
    sortDirection.value = sortDirection.value === 'asc' ? 'desc' : 'asc'
  } else {
    // 如果点击的是新字段，设置为升序
    sortField.value = field
    sortDirection.value = 'asc'
  }
  currentPage.value = 1 // 重置到第一页
}

// 获取排序图标
const getSortIcon = (field) => {
  if (sortField.value !== field) {
    return 'fas fa-sort text-gray-400'
  }
  return sortDirection.value === 'asc' ? 'fas fa-sort-up text-blue-600' : 'fas fa-sort-down text-blue-600'
}

// 处理多个分子数据（SDF文件上传）
const filteredItems = computed(() => {
  if (dataSource.value !== 'file' || !resultData.value?.items) return []
  
  let items = [...resultData.value.items]
  
  // 搜索过滤
  if (searchQuery.value.trim()) {
    const query = searchQuery.value.toLowerCase()
    items = items.filter(item => 
      item.smiles.toLowerCase().includes(query)
    )
  }
  
  // 排序
  if (sortField.value) {
    items.sort((a, b) => {
      let valueA, valueB
      
      if (sortField.value === 'smiles') {
        valueA = a.smiles
        valueB = b.smiles
      } else if (sortField.value === 'matches') {
        valueA = a.matches[selectedRule.value] || 0
        valueB = b.matches[selectedRule.value] || 0
      } else {
        // 其他字段从metrics中获取
        valueA = a.metrics[selectedRule.value]?.[sortField.value] || 0
        valueB = b.metrics[selectedRule.value]?.[sortField.value] || 0
      }
      
      // 处理字符串和数字比较
      if (typeof valueA === 'string' && typeof valueB === 'string') {
        valueA = valueA.toLowerCase()
        valueB = valueB.toLowerCase()
      }
      
      if (valueA < valueB) {
        return sortDirection.value === 'asc' ? -1 : 1
      }
      if (valueA > valueB) {
        return sortDirection.value === 'asc' ? 1 : -1
      }
      return 0
    })
  }
  
  return items
})

const paginatedItems = computed(() => {
  const start = (currentPage.value - 1) * pageSize.value
  const end = start + pageSize.value
  return filteredItems.value.slice(start, end)
})

const totalPages = computed(() => {
  return Math.ceil(filteredItems.value.length / pageSize.value)
})

const totalItems = computed(() => {
  return filteredItems.value.length
})

const showingStart = computed(() => {
  return Math.min((currentPage.value - 1) * pageSize.value + 1, totalItems.value)
})

const showingEnd = computed(() => {
  return Math.min(currentPage.value * pageSize.value, totalItems.value)
})

// 页面初始化
onMounted(() => {
  // 清理过期数据
  cleanupExpiredData()
  
  // 从sessionStorage中获取结果数据
  const resultId = route.query.resultId
  
  if (resultId) {
    const parsedData = getResultData(resultId)
    
    if (parsedData) {
      // 设置数据
      resultData.value = parsedData.data
      selectedRule.value = parsedData.rule || 'Lipinski'
      dataSource.value = parsedData.source || 'smiles'
      
      // 如果是文件上传结果，重置分页
      if (dataSource.value === 'file') {
        currentPage.value = 1
        searchQuery.value = ''
      }
      
      // 清理URL，移除resultId参数
      router.replace({ path: '/druglikeness-result' })
    } else {
      // 数据不存在或已过期
      router.push('/druglikeness-evaluation')
    }
  } else {
    // 如果没有resultId，重定向回评估页面
    router.push('/druglikeness-evaluation')
  }
})

// 注意：cleanupExpiredData 函数现在从 storage.js 导入

// 返回评估页面
const goBack = () => {
  // 清理当前结果数据
  const resultId = route.query.resultId
  if (resultId) {
    removeResultData(resultId)
  }
  
  // 清理过期数据
  cleanupExpiredData()
  
  router.push('/druglikeness-evaluation')
}

// 分页相关函数
const goToPage = (page) => {
  if (page >= 1 && page <= totalPages.value) {
    currentPage.value = page
  }
}

const goToFirstPage = () => {
  currentPage.value = 1
}

const goToLastPage = () => {
  currentPage.value = totalPages.value
}

// 搜索重置
const clearSearch = () => {
  searchQuery.value = ''
  currentPage.value = 1
}

// 清除排序
const clearSort = () => {
  sortField.value = ''
  sortDirection.value = 'asc'
  currentPage.value = 1
}

// 导出CSV功能
const exportToCSV = () => {
  if (!resultData.value) {
    alert('没有可导出的数据')
    return
  }

  // 准备导出数据
  const exportData = []
  
  // 添加标题行
  const headers = ['Molecule', ...Object.values(tableHeaders.value), 'Matches (%)']
  exportData.push(headers)
  
  if (dataSource.value === 'smiles') {
    // 单个分子数据
    const dataRow = [
      resultData.value.smiles || 'N/A',
      ...Object.values(currentMetrics.value).map(value => 
        typeof value === 'number' ? (Math.floor(value * 100) / 100).toString() : value
      ),
      currentMatchPercentage.value + '%'
    ]
    exportData.push(dataRow)
  } else if (dataSource.value === 'file') {
    // 多个分子数据（导出所有数据，不仅仅是当前页）
    filteredItems.value.forEach(item => {
      const dataRow = [
        item.smiles,
        ...Object.values(item.metrics[selectedRule.value]).map(value => 
          typeof value === 'number' ? (Math.floor(value * 100) / 100).toString() : value
        ),
        (item.matches[selectedRule.value] * 100).toFixed(2) + '%'
      ]
      exportData.push(dataRow)
    })
  }
  
  // 创建工作表
  const ws = XLSX.utils.aoa_to_sheet(exportData)
  
  // 创建工作簿
  const wb = XLSX.utils.book_new()
  XLSX.utils.book_append_sheet(wb, ws, 'Druglikeness Results')
  
  // 生成文件名
  const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-')
  const sourceType = dataSource.value === 'file' ? 'batch' : 'single'
  const filename = `druglikeness_${selectedRule.value}_${sourceType}_${timestamp}.csv`
  
  // 导出文件
  XLSX.writeFile(wb, filename)
}
</script>

<style scoped>
/* 自定义样式 */
.table-cell {
  padding-left: 0.75rem;
  padding-right: 0.75rem;
  padding-top: 0.5rem;
  padding-bottom: 0.5rem;
  font-size: 0.875rem;
  line-height: 1.25rem;
}
</style>