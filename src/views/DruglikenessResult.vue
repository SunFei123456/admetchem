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
          <span class="font-medium">Result</span>
        </div>


        <!-- 结果表格 -->
        <div class="overflow-x-auto">
          <table class="w-full">
            <thead class="bg-gray-50">
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
              <!-- 单个分子结果（SMILES输入） -->
              <tr v-if="dataSource === 'smiles'" class="border-b border-gray-200 hover:bg-gray-50">
                <td class="px-4 py-3">
                  <div class="flex items-center">
                    <span class="font-mono text-sm break-all">{{ resultData?.smiles || 'N/A' }}</span>
                  </div>
                </td>
                <td v-for="rowValue in getRowValues" :key="rowValue.key" class="px-4 py-3 text-sm whitespace-nowrap">
                  {{ rowValue.value }}
                </td>
              </tr>

              <!-- 无数据提示 -->
              <tr v-if="!resultData">
                <td :colspan="tableHeaders.length + 1" class="px-4 py-8 text-center text-gray-500">
                  暂无数据
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <!-- 分页信息 -->
        <div class="p-4 border-t border-gray-200 flex justify-between items-center">
          <div class="text-sm text-gray-600">
            Showing 1 entry
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
const dataSource = ref('smiles') // 'smiles' 或 'file'

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

// 计算属性：动态生成表头 --> 数据来源于从接口返回中进行提取
const tableHeaders = computed(() => {
  const headers = []

  // 添加分子性质列
  if (resultData.value?.molecular_properties) {
    const props = resultData.value.molecular_properties
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
  if (resultData.value?.druglikeness_rules?.matches) {
    const matches = resultData.value.druglikeness_rules.matches
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

// 根据KEY 获取单行数据的值
const getRowValues = computed(() => {
  if (!resultData.value) return []

  const values = []

  tableHeaders.value.forEach(header => {
    if (header.type === 'property') {
      // 分子性质
      const value = resultData.value.molecular_properties?.[header.key]
      values.push({
        key: header.key,
        value: value != null ? (typeof value === 'number' ? value.toFixed(4) : value) : 'N/A'
      })
    } else if (header.type === 'match') {
      // 规则匹配度
      const matchValue = resultData.value.druglikeness_rules?.matches?.[header.key]
      values.push({
        key: header.key,
        value: matchValue != null ? (matchValue * 100).toFixed(2) + '%' : 'N/A'
      })
    }
  })

  return values
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
      selectedItems.value = parsedData.selectedItems || []
      dataSource.value = parsedData.source || 'smiles'

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
  if (!resultData.value) {
    alert('没有可导出的数据')
    return
  }

  // 准备导出数据
  const exportData = []

  // 添加标题行
  const headers = ['Molecule', ...tableHeaders.value.map(h => h.label)]
  exportData.push(headers)

  // 添加数据行
  const dataRow = [
    resultData.value.smiles || 'N/A',
    ...getRowValues.value.map(v => v.value)
  ]
  exportData.push(dataRow)

  // 创建工作表
  const ws = XLSX.utils.aoa_to_sheet(exportData)

  // 创建工作簿
  const wb = XLSX.utils.book_new()
  XLSX.utils.book_append_sheet(wb, ws, 'Evaluation Results')

  // 生成文件名
  const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-')
  const itemsCount = selectedItems.value.length
  const filename = `evaluation_results_${itemsCount}items_${timestamp}.csv`

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