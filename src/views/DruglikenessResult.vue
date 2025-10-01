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

      <!-- 结果展示表格 -->
      <div class="bg-white rounded-lg border border-slate-200">
        <!-- 标题栏 -->
        <div class="bg-green-500 text-white px-4 py-3 flex items-center">
          <i class="fas fa-check-circle mr-2"></i>
          <span class="font-medium">Result</span>
        </div>

        <!-- 结果表格 -->
        <div class="overflow-x-auto">
          <table class="w-full">
            <thead class="bg-gray-50">
              <!-- 表格头 数据来源于从接口返回中进行提取 -->
              <tr>
                <th class="px-4 py-3 text-left text-sm font-medium text-gray-700 border-b min-w-[300px]">
                  Molecule
                </th>
                <th v-for="header in tableHeaders" :key="header.key"
                  class="px-4 py-3 text-left text-sm font-medium text-gray-700 border-b whitespace-nowrap">
                  {{ header.label }}
                </th>
              </tr>
            </thead>
            <tbody>
              <!-- 当前页的所有分子 -->
              <tr v-for="(molecule, index) in currentPageData" :key="index"
                class="border-b border-gray-200 hover:bg-gray-50">
                <!-- 分子列 -->
                <td class="px-4 py-3">
                  <div class="flex items-center">
                    <span class="font-mono text-sm break-all">{{ molecule?.smiles || 'N/A' }}</span>
                  </div>
                </td>
                <!-- 分子性质列 -->
                <td v-for="rowValue in getRowValuesForMolecule(molecule)" :key="rowValue.key"
                  class="px-4 py-3 text-sm whitespace-nowrap">
                  {{ rowValue.value }}
                </td>
              </tr>

              <!-- 无数据提示 -->
              <tr v-if="currentPageData.length === 0">
                <td :colspan="tableHeaders.length + 1" class="px-4 py-8 text-center text-gray-500">
                  暂无数据
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <!-- 分页信息和分页器 -->
        <div class="p-4 border-t border-gray-200">
          <div class="flex justify-between items-center">
            <!-- 左侧：显示信息 -->
            <div class="text-sm text-gray-600">
              <span v-if="dataSource === 'file' && moleculesList.length > 0">
                显示 {{ (currentPage - 1) * pageSize + 1 }} - {{ Math.min(currentPage * pageSize, moleculesList.length) }}
                条，共 {{ moleculesList.length }} 条
              </span>
              <span v-else>
                Showing 1 entry
              </span>
            </div>

            <!-- 右侧：分页器 -->
            <div v-if="dataSource === 'file' && totalPages > 1" class="flex items-center space-x-2">
              <!-- 上一页按钮 -->
              <button @click="previousPage" :disabled="currentPage === 1"
                class="px-3 py-1 text-sm border rounded hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed">
                上一页
              </button>

              <!-- 页码按钮 -->
              <template v-for="page in totalPages" :key="page">
                <button v-if="page === 1 || page === totalPages || (page >= currentPage - 2 && page <= currentPage + 2)"
                  @click="goToPage(page)" :class="[
                    'px-3 py-1 text-sm border rounded',
                    page === currentPage
                      ? 'bg-blue-600 text-white border-blue-600'
                      : 'hover:bg-gray-100'
                  ]">
                  {{ page }}
                </button>
                <span v-else-if="page === currentPage - 3 || page === currentPage + 3"
                  class="px-2 text-gray-500">...</span>
              </template>

              <!-- 下一页按钮 -->
              <button @click="nextPage" :disabled="currentPage === totalPages"
                class="px-3 py-1 text-sm border rounded hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed">
                下一页
              </button>
            </div>
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
import * as XLSX from 'xlsx'
import { getResultData, removeResultData, cleanupExpiredData } from '../utils/storage.js'

const route = useRoute()
const router = useRouter()



// 结果数据
const resultData = ref(null)
const selectedItems = ref([]) // 用户选择的评估项目
const dataSource = ref('smiles') // 数据来源: 'smiles' 或 'file'
const moleculesList = ref([]) // 多分子文件中的分子列表

// 分页相关
const currentPage = ref(1) // 当前页码
const pageSize = ref(10) // 每页显示条数

// 分子性质名称映射
const propertyNameMapping = {
  'QED': 'QED',
  'SAscore': 'SAscore',
  'Fsp3': 'Fsp3',
  'MCE18': 'MCE18',
  'NPscore': 'NPscore'
}

// 规则名称映射（用于matches列）
const ruleNameMapping = {
  'Lipinski': 'Lipinski Match',
  'Ghose': 'Ghose Match',
  'Oprea': 'Oprea Match',
  'Veber': 'Veber Match',
  'Varma': 'Varma Match'
}

// 计算属性：总页数
const totalPages = computed(() => {
  if (dataSource.value === 'file' && moleculesList.value.length > 0) {
    return Math.ceil(moleculesList.value.length / pageSize.value)
  }
  return 1
})

// 计算属性：当前页显示的数据
const currentPageData = computed(() => {
  if (dataSource.value === 'file' && moleculesList.value.length > 0) {
    const start = (currentPage.value - 1) * pageSize.value
    const end = start + pageSize.value
    return moleculesList.value.slice(start, end)
  } else if (resultData.value) {
    // SMILES输入，只有一条数据
    return [resultData.value]
  }
  return []
})

// 计算属性：动态生成表头 --> 数据来源于从接口返回中进行提取
const tableHeaders = computed(() => {
  const headers = []

  // 使用第一条数据或resultData作为参考
  const refData = currentPageData.value.length > 0 ? currentPageData.value[0] : resultData.value

  // 添加分子性质列
  if (refData?.molecular_properties) {
    const props = refData.molecular_properties
    // 遍历所有分子性质，只显示值，不显示status
    Object.keys(props).forEach(key => {
      if (!key.endsWith('_status') && !key.endsWith('_error')) {
        headers.push({
          key: key,
          label: propertyNameMapping[key] || key,
          type: 'property'
        })
      }
    })
  }

  // 添加规则matches列
  if (refData?.druglikeness_rules?.matches) {
    const matches = refData.druglikeness_rules.matches
    Object.keys(matches).forEach(ruleName => {
      headers.push({
        key: ruleName,
        label: ruleNameMapping[ruleName] || `${ruleName} Match`,
        type: 'match'
      })
    })
  }

  return headers
})

// 为指定分子获取行值
const getRowValuesForMolecule = (molecule) => {
  if (!molecule) return []

  const values = []

  tableHeaders.value.forEach(header => {
    if (header.type === 'property') {
      // 分子性质
      const value = molecule.molecular_properties?.[header.key]
      values.push({
        key: header.key,
        value: value != null ? (typeof value === 'number' ? value.toFixed(4) : value) : 'N/A'
      })
    } else if (header.type === 'match') {
      // 规则匹配度
      const matchValue = molecule.druglikeness_rules?.matches?.[header.key]
      values.push({
        key: header.key,
        value: matchValue != null ? (matchValue * 100).toFixed(2) + '%' : 'N/A'
      })
    }
  })

  return values
}


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
      selectedItems.value = parsedData.selectedItems || []
      dataSource.value = parsedData.source || 'smiles'

      // 处理不同数据源的数据结构
      if (dataSource.value === 'file' && parsedData.data.items) {
        // SDF文件数据：多分子结构
        moleculesList.value = parsedData.data.items
        currentPage.value = 1
      } else {
        // SMILES数据：单分子结构
        resultData.value = parsedData.data
        moleculesList.value = []
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

// 分页控制函数
const goToPage = (page) => {
  if (page >= 1 && page <= totalPages.value) {
    currentPage.value = page
  }
}

const previousPage = () => {
  if (currentPage.value > 1) {
    currentPage.value--
  }
}

const nextPage = () => {
  if (currentPage.value < totalPages.value) {
    currentPage.value++
  }
}

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

// 导出CSV功能
const exportToCSV = () => {
  if (currentPageData.value.length === 0) {
    alert('没有可导出的数据')
    return
  }

  // 准备导出数据
  const exportData = []

  // 添加标题行
  const headers = ['Molecule', ...tableHeaders.value.map(h => h.label)]
  exportData.push(headers)

  // 根据数据源决定导出内容
  const dataToExport = dataSource.value === 'file' && moleculesList.value.length > 0
    ? moleculesList.value
    : currentPageData.value

  // 导出所有分子
  dataToExport.forEach((molecule) => {
    const rowValues = getRowValuesForMolecule(molecule)
    const dataRow = [
      molecule.smiles || 'N/A',
      ...rowValues.map(rv => rv.value)
    ]
    exportData.push(dataRow)
  })

  // 创建工作表
  const ws = XLSX.utils.aoa_to_sheet(exportData)

  // 创建工作簿
  const wb = XLSX.utils.book_new()
  XLSX.utils.book_append_sheet(wb, ws, 'Evaluation Results')

  // 生成文件名
  const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-')
  const itemsCount = selectedItems.value.length
  const moleculeCount = dataToExport.length
  const filename = `evaluation_results_${itemsCount}items_${moleculeCount}molecules_${timestamp}.csv`

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