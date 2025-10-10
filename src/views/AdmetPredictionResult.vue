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

            <!-- 搜索面板（仅在有多个分子时显示） -->
            <div v-if="dataSource === 'file' && moleculesList.length > 0"
                class="bg-white rounded-lg border border-slate-200 p-4 mb-6">

                <!-- 属性搜索 -->
                <div class="mb-4">
                    <h3 class="text-base font-bold text-gray-800 mb-3 flex items-center">
                        <i class="fas fa-filter mr-2 text-blue-600"></i>
                        <span>Attribute Filtering</span>
                    </h3>
                    <div class="flex items-end space-x-4">
                        <div class="flex-shrink-0" style="width: 240px;">
                            <label class="block text-xs font-semibold text-gray-600 mb-1.5 uppercase tracking-wide">
                                Property
                            </label>
                            <select v-model="searchField"
                                class="w-full px-4 py-2.5 text-sm font-medium text-gray-700 bg-white border-2 border-gray-300 rounded-lg shadow-sm appearance-none cursor-pointer transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 hover:border-gray-400"
                                style="background-image: url('data:image/svg+xml;charset=UTF-8,%3csvg xmlns=%27http://www.w3.org/2000/svg%27 viewBox=%270 0 24 24%27 fill=%27none%27 stroke=%27currentColor%27 stroke-width=%272%27 stroke-linecap=%27round%27 stroke-linejoin=%27round%27%3e%3cpolyline points=%276 9 12 15 18 9%27%3e%3c/polyline%3e%3c/svg%3e'); background-repeat: no-repeat; background-position: right 0.75rem center; background-size: 1.25em 1.25em; padding-right: 2.5rem;">
                                <option value="" class="text-gray-400 text-sm">Please select an attribute</option>
                                <optgroup v-for="group in propertyGroups" :key="group.label" :label="group.label"
                                    class="font-semibold text-gray-800 bg-gray-50">
                                    <option v-for="opt in group.options" :key="opt.value" :value="opt.value"
                                        class="text-gray-700 font-normal py-1">
                                        {{ opt.label }}
                                    </option>
                                </optgroup>
                            </select>
                        </div>

                        <!-- 分类属性搜索（0或1） -->
                        <div v-if="searchField && isClassificationProperty(searchField)" class="flex-1">
                            <label class="block text-xs font-semibold text-gray-600 mb-1.5 uppercase tracking-wide">
                                Classification Value
                            </label>
                            <select v-model="searchClassValue"
                                class="w-full px-4 py-2.5 text-sm font-medium text-gray-700 bg-white border-2 border-gray-300 rounded-lg shadow-sm appearance-none cursor-pointer transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 hover:border-gray-400"
                                style="background-image: url('data:image/svg+xml;charset=UTF-8,%3csvg xmlns=%27http://www.w3.org/2000/svg%27 viewBox=%270 0 24 24%27 fill=%27none%27 stroke=%27currentColor%27 stroke-width=%272%27 stroke-linecap=%27round%27 stroke-linejoin=%27round%27%3e%3cpolyline points=%276 9 12 15 18 9%27%3e%3c/polyline%3e%3c/svg%3e'); background-repeat: no-repeat; background-position: right 0.75rem center; background-size: 1.25em 1.25em; padding-right: 2.5rem;">
                                <option value="" class="text-gray-500">All</option>
                                <option value="1" class="text-green-600 font-medium">1 (Positive)</option>
                                <option value="0" class="text-red-600 font-medium">0 (Negative)</option>
                            </select>
                        </div>

                        <!-- 回归属性搜索（范围） -->
                        <div v-if="searchField && !isClassificationProperty(searchField)" class="flex-1">
                            <label class="block text-xs font-semibold text-gray-600 mb-1.5 uppercase tracking-wide">
                                Minimum Value
                            </label>
                            <input v-model="searchMinValue" type="number" step="any" placeholder="Enter minimum"
                                class="w-full px-4 py-2.5 text-sm font-medium text-gray-700 bg-white border-2 border-gray-300 rounded-lg shadow-sm transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 hover:border-gray-400 placeholder-gray-400">
                        </div>
                        <div v-if="searchField && !isClassificationProperty(searchField)" class="flex-1">
                            <label class="block text-xs font-semibold text-gray-600 mb-1.5 uppercase tracking-wide">
                                Maximum Value
                            </label>
                            <input v-model="searchMaxValue" type="number" step="any" placeholder="Enter maximum"
                                class="w-full px-4 py-2.5 text-sm font-medium text-gray-700 bg-white border-2 border-gray-300 rounded-lg shadow-sm transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 hover:border-gray-400 placeholder-gray-400">
                        </div>
                    </div>
                </div>

                <!-- 操作按钮 -->
                <div class="flex items-center justify-between mt-4">
                    <div class="flex space-x-3">
                        <button @click="applySearch" :disabled="!searchField"
                            class="px-8 py-2.5 bg-gradient-to-r from-blue-600 to-blue-700 text-white font-semibold rounded-lg shadow-md hover:from-blue-700 hover:to-blue-800 disabled:from-gray-400 disabled:to-gray-400 disabled:cursor-not-allowed transition-all duration-200 transform hover:scale-105 disabled:transform-none">
                            <i class="fas fa-search mr-2"></i>
                            Search
                        </button>
                        <button @click="clearSearch"
                            class="px-8 py-2.5 bg-gradient-to-r from-gray-500 to-gray-600 text-white font-semibold rounded-lg shadow-md hover:from-gray-600 hover:to-gray-700 transition-all duration-200 transform hover:scale-105">
                            <i class="fas fa-times mr-2"></i>
                            Clear
                        </button>
                    </div>

                    <!-- 搜索结果提示 -->
                    <div v-if="appliedSearchField && filteredMolecules.length < moleculesList.length"
                        class="text-sm font-medium text-blue-700 bg-blue-50 px-4 py-2 rounded-lg border border-blue-200">
                        <i class="fas fa-info-circle mr-2"></i>
                        Found <span class="font-bold">{{ filteredMolecules.length }}</span> results
                        <span class="text-gray-600">(out of {{ moleculesList.length }} total)</span>
                    </div>
                </div>
            </div>

            <!-- 结果展示表格 -->
            <div class="bg-white rounded-lg border border-slate-200">
                <!-- 标题栏 -->
                <div class="bg-green-500 text-white px-4 py-3 flex items-center">
                    <i class="fas fa-check-circle mr-2"></i>
                    <span class="font-medium">ADMET Prediction Result</span>
                </div>

                <!-- 结果表格 -->
                <div class="overflow-x-auto">
                    <table class="w-full">
                        <thead class="bg-gray-50">
                            <tr>
                                <th
                                    class="px-4 py-3 text-left text-sm font-medium text-gray-700 border-b min-w-[300px]">
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
                                <td class="px-4 py-3 cursor-pointer hover:bg-blue-50 transition-colors duration-200"
                                    @click="goToMoleculeDetail(molecule)">
                                    <div class="flex items-center">
                                        <span
                                            class="font-mono text-sm break-all text-blue-600 hover:text-blue-800 hover:underline">
                                            {{ molecule?.smiles || 'N/A' }}
                                        </span>
                                    </div>
                                </td>
                                <!-- ADMET 属性值列 -->
                                <td v-for="value in getPropertyValuesForMolecule(molecule)" :key="value.key"
                                    class="px-4 py-3 text-sm whitespace-nowrap">
                                    {{ value.displayValue }}
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
                            <span v-if="dataSource === 'file' && filteredMolecules.length > 0">
                                Showing {{ (currentPage - 1) * pageSize + 1 }} - {{ Math.min(currentPage * pageSize,
                                    filteredMolecules.length) }}
                                results (out of {{ filteredMolecules.length }} total)
                                <span v-if="appliedSearchField && filteredMolecules.length < moleculesList.length"
                                    class="text-blue-600">
                                    (filtered, total {{ moleculesList.length }})
                                </span>
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
                                Pre
                            </button>

                            <!-- 页码按钮 -->
                            <template v-for="page in totalPages" :key="page">
                                <button
                                    v-if="page === 1 || page === totalPages || (page >= currentPage - 2 && page <= currentPage + 2)"
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
                                Next
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
                    Export CSV
                </button>
                <button @click="goBack"
                    class="bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-700 transition duration-200">
                    <i class="fas fa-arrow-left mr-2"></i>
                    Back to prediction page
                </button>
            </div>
        </main>
    </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import * as XLSX from 'xlsx'

const router = useRouter()

// 结果数据
const resultData = ref(null)
const dataSource = ref('smiles') // 数据来源: 'smiles' 或 'file'
const moleculesList = ref([]) // 多分子文件中的分子列表

// 分页相关
const currentPage = ref(1)
const pageSize = ref(10)

// 搜索相关 - 临时输入值
const searchField = ref('') // 搜索属性
const searchMinValue = ref('') // 最小值（回归）
const searchMaxValue = ref('') // 最大值（回归）
const searchClassValue = ref('') // 分类值（0或1）

// 应用的搜索条件
const appliedSearchField = ref('')
const appliedSearchMin = ref('')
const appliedSearchMax = ref('')
const appliedSearchClassValue = ref('')

// 分类属性列表（这些属性搜索0或1）
const classificationProperties = [
    'cyp1a2', 'cyp2c9', 'cyp2c9-sub', 'cyp2c19', 'cyp2d6', 'cyp2d6-sub',
    'cyp3a4', 'cyp3a4-sub', 'herg', 'pgp', 'ames', 'bbb', 'bioavailability',
    'dili', 'hia', 'skinSen', 'nr-ar-lbd', 'nr-ar', 'nr-ahr', 'nr-aromatase',
    'nr-er', 'nr-er-lbd', 'nr-ppar-gamma', 'sr-are', 'sr-atad5', 'sr-hse',
    'sr-mmp', 'sr-p53', 'pampa'
]

// 属性分组选项
const propertyGroups = ref([
    {
        label: 'Biophysics',
        options: [
            { value: 'cyp1a2', label: 'CYP1A2-inh' },
            { value: 'cyp2c9', label: 'CYP2C9-inh' },
            { value: 'cyp2c9-sub', label: 'CYP2C9-sub' },
            { value: 'cyp2c19', label: 'CYP2C19-inh' },
            { value: 'cyp2d6', label: 'CYP2D6-inh' },
            { value: 'cyp2d6-sub', label: 'CYP2D6-sub' },
            { value: 'cyp3a4', label: 'CYP3A4-inh' },
            { value: 'cyp3a4-sub', label: 'CYP3A4-sub' },
            { value: 'herg', label: 'hERG blockers' },
            { value: 'pgp', label: 'Pgp-inh' }
        ]
    },
    {
        label: 'Physical Chemistry',
        options: [
            { value: 'logS', label: 'LogS' },
            { value: 'logP', label: 'LogP' },
            { value: 'logD', label: 'LogD' },
            { value: 'hydration', label: 'Hydration Free Energy' },
            { value: 'pampa', label: 'PAMPA' }
        ]
    },
    {
        label: 'Physiology',
        options: [
            { value: 'ames', label: 'Ames' },
            { value: 'bbb', label: 'BBB' },
            { value: 'bioavailability', label: 'Bioavailability' },
            { value: 'caco2', label: 'Caco-2' },
            { value: 'clearance', label: 'CL' },
            { value: 'dili', label: 'DILI' },
            { value: 'halflife', label: 'Drug Half-Life' },
            { value: 'hia', label: 'HIA' },
            { value: 'ppbr', label: 'PPBR' },
            { value: 'skinSen', label: 'SkinSen' },
            { value: 'nr-ar-lbd', label: 'NR-AR-LBD' },
            { value: 'nr-ar', label: 'NR-AR' },
            { value: 'nr-ahr', label: 'NR-AhR' },
            { value: 'nr-aromatase', label: 'NR-Aromatase' },
            { value: 'nr-er', label: 'NR-ER' },
            { value: 'nr-er-lbd', label: 'NR-ER-LBD' },
            { value: 'nr-ppar-gamma', label: 'NR-PPAR-gamma' },
            { value: 'sr-are', label: 'SR-ARE' },
            { value: 'sr-atad5', label: 'SR-ATAD5' },
            { value: 'sr-hse', label: 'SR-HSE' },
            { value: 'sr-mmp', label: 'SR-MMP' },
            { value: 'sr-p53', label: 'SR-p53' },
            { value: 'vdss', label: 'VDss' }
        ]
    }
])

// 页面初始化
onMounted(() => {
    // 从 router state 中获取数据
    const state = history.state

    if (state && state.result) {
        // 检查数据结构
        if (state.result.results && Array.isArray(state.result.results)) {
            // SDF 文件批量数据：{ results: [...], count: 100 }
            moleculesList.value = state.result.results
            dataSource.value = 'file'
            currentPage.value = 1
        } else if (state.result.smiles && state.result.predictions) {
            // 单个 SMILES 数据：{ smiles: "...", predictions: {...} }
            resultData.value = state.result
            dataSource.value = 'smiles'
            moleculesList.value = []
        } else {
            // 未知格式，重定向
            router.push('/admet-prediction')
        }
    } else {
        // 如果没有数据，重定向回预测页面
        router.push('/admet-prediction')
    }
})

// 判断是否为分类属性
const isClassificationProperty = (propId) => {
    return classificationProperties.includes(propId)
}

// 过滤后的分子列表
const filteredMolecules = computed(() => {
    if (dataSource.value !== 'file' || moleculesList.value.length === 0) {
        return resultData.value ? [resultData.value] : []
    }

    // 如果没有应用搜索条件，返回所有数据
    if (!appliedSearchField.value) {
        return moleculesList.value
    }

    // 应用搜索过滤
    return moleculesList.value.filter(molecule => {
        const prediction = molecule.predictions?.[appliedSearchField.value]
        if (!prediction) return false

        // 分类属性过滤（0或1）
        if (isClassificationProperty(appliedSearchField.value)) {
            if (!appliedSearchClassValue.value) return true // 未选择具体值，显示全部

            // 获取显示值（0或1）
            let displayValue = null
            if (prediction.symbol) {
                displayValue = prediction.symbol.includes('-') ? '0' : '1'
            } else if (prediction.probability != null) {
                displayValue = prediction.probability.toFixed(2)
            }

            return displayValue === appliedSearchClassValue.value
        }

        // 回归属性过滤（范围）
        let value = null
        if (prediction.value != null) {
            value = prediction.value
        } else if (prediction.probability != null) {
            value = prediction.probability
        }

        if (value == null || typeof value !== 'number') return false

        const min = appliedSearchMin.value ? parseFloat(appliedSearchMin.value) : -Infinity
        const max = appliedSearchMax.value ? parseFloat(appliedSearchMax.value) : Infinity

        return value >= min && value <= max
    })
})

// 计算属性：总页数
const totalPages = computed(() => {
    return Math.ceil(filteredMolecules.value.length / pageSize.value) || 1
})

// 计算属性：当前页显示的数据
const currentPageData = computed(() => {
    const start = (currentPage.value - 1) * pageSize.value
    const end = start + pageSize.value
    return filteredMolecules.value.slice(start, end)
})

// 计算属性：动态生成表头
const tableHeaders = computed(() => {
    const refData = currentPageData.value.length > 0 ? currentPageData.value[0] : resultData.value

    if (!refData?.predictions) return []

    const headers = []
    const predictions = refData.predictions

    // 遍历所有预测属性
    Object.keys(predictions).forEach(key => {
        const prediction = predictions[key]
        headers.push({
            key: key,
            label: prediction.name || key
        })
    })

    return headers
})

// 为指定分子获取属性值
const getPropertyValuesForMolecule = (molecule) => {
    if (!molecule?.predictions) return []

    const values = []
    const predictions = molecule.predictions

    Object.keys(predictions).forEach(key => {
        const prediction = predictions[key]
        let displayValue = 'N/A'

        if (prediction.symbol) {
            // 如果含有 symbol
            if (prediction.symbol.includes('-')) {
                // 包含 - 号，显示 0
                displayValue = '0'
            } else if (prediction.symbol.includes('+')) {
                // 包含 + 号，显示 1
                displayValue = '1'
            }
        } else if (prediction.probability != null) {
            // 不含 symbol，显示 probability 的小数点后两位
            displayValue = prediction.probability.toFixed(2)
        } else if (prediction.value != null) {
            // 回归类型，显示 value 的小数点后两位
            displayValue = prediction.value.toFixed(2)
        }

        values.push({
            key: key,
            displayValue: displayValue
        })
    })

    return values
}

// 搜索控制函数
const applySearch = () => {
    // 将临时输入值应用到实际的过滤条件
    appliedSearchField.value = searchField.value

    if (isClassificationProperty(searchField.value)) {
        // 分类属性
        appliedSearchClassValue.value = searchClassValue.value
        appliedSearchMin.value = ''
        appliedSearchMax.value = ''
    } else {
        // 回归属性
        appliedSearchMin.value = searchMinValue.value
        appliedSearchMax.value = searchMaxValue.value
        appliedSearchClassValue.value = ''
    }

    // 应用搜索后回到第一页
    currentPage.value = 1
}

const clearSearch = () => {
    // 清空临时输入值
    searchField.value = ''
    searchMinValue.value = ''
    searchMaxValue.value = ''
    searchClassValue.value = ''

    // 清空应用的搜索条件
    appliedSearchField.value = ''
    appliedSearchMin.value = ''
    appliedSearchMax.value = ''
    appliedSearchClassValue.value = ''

    currentPage.value = 1
}

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

// 返回预测页面
const goBack = () => {
    router.push('/admet-prediction')
}

// 跳转到分子详情页面
const goToMoleculeDetail = (molecule) => {
    // 将分子数据存储到 sessionStorage，避免 DataCloneError
    const moleculeData = JSON.parse(JSON.stringify(molecule))
    sessionStorage.setItem('currentMolecule', JSON.stringify(moleculeData))

    router.push({
        name: 'molecule-detail'
    })
}

// 导出CSV功能
const exportToCSV = () => {
    if (currentPageData.value.length === 0) {
        alert('No data to export')
        return
    }

    // 准备导出数据
    const exportData = []

    // 添加标题行
    const headers = ['Molecule', ...tableHeaders.value.map(h => h.label)]
    exportData.push(headers)

    // 根据数据源决定导出内容
    const dataToExport = dataSource.value === 'file' && filteredMolecules.value.length > 0
        ? filteredMolecules.value
        : currentPageData.value

    // 导出所有分子
    dataToExport.forEach((molecule) => {
        const rowValues = getPropertyValuesForMolecule(molecule)
        const dataRow = [
            molecule.smiles || 'N/A',
            ...rowValues.map(rv => rv.displayValue)
        ]
        exportData.push(dataRow)
    })

    // 创建工作表
    const ws = XLSX.utils.aoa_to_sheet(exportData)

    // 创建工作簿
    const wb = XLSX.utils.book_new()
    XLSX.utils.book_append_sheet(wb, ws, 'ADMET Prediction Results')

    // 生成文件名
    const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-')
    const propertyCount = tableHeaders.value.length
    const moleculeCount = dataToExport.length
    const filtered = appliedSearchField.value ? '_filtered' : ''
    const filename = `admet_prediction_${propertyCount}properties_${moleculeCount}molecules${filtered}_${timestamp}.csv`

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
