<template>
    <div class="bg-gray-50 min-h-screen">
   

        <main class="max-w-7xl mx-auto px-4 py-6">
            <div class="grid lg:grid-cols-3 gap-6">

                <!-- 左侧：规则选择和详细信息 -->
                <div class="lg:col-span-1 space-y-6">

                    <!-- 点击选择和计算 -->
                    <div class="bg-white rounded-lg border border-slate-200">
                        <div class="bg-gray-200 px-4 py-3    flex items-center">
                            <i class="fas fa-hand-pointer text-gray-600 mr-2"></i>
                            <span class="font-medium text-gray-700">Click to select & calculate</span>
                        </div>
                        <div class="p-4 space-y-3">
                            <div v-for="rule in rules" :key="rule.id" class="flex items-center">
                                <input type="radio" :id="rule.id" :value="rule.id" v-model="selectedRule"
                                    class="mr-3 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300">
                                <label :for="rule.id" :class="[
                                    'cursor-pointer text-sm transition-colors duration-200',
                                    selectedRule === rule.id 
                                        ? 'text-blue-800 font-semibold' 
                                        : 'text-blue-600 hover:text-blue-800'
                                ]">
                                    {{ rule.name }}
                                </label>
                            </div>
                        </div>
                    </div>

                    <!-- 详细信息 -->
                    <div class="bg-white rounded-lg border border-slate-200">
                        <div class="bg-gray-200 px-4 py-3    flex items-center">
                            <i class="fas fa-info-circle text-gray-600 mr-2"></i>
                            <span class="font-medium text-gray-700">Detailed information</span>
                        </div>
                        <div class="p-4">

                            <!-- 模型性能表 -->
                            <div class="mb-6">
                                <h4 class="font-semibold mb-3 text-left">Rule Details</h4>
                                <div class="overflow-x-auto">
                                    <table class="w-full text-sm border border-gray-300">
                                        <tbody>
                                            <tr v-for="(item, index) in modelPerformance" :key="index"
                                                class="border border-gray-200">
                                                <td
                                                    class="px-3 py-2 bg-gray-50 font-medium text-gray-700 border-r border-gray-300">
                                                    {{ item.label }}</td>
                                                <td class="px-3 py-2 text-gray-900 whitespace-pre-line">{{ item.value }}</td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </div>
                            </div>


                        </div>
                    </div>
                </div>

                <!-- 右侧：输入方式 -->
                <div class="lg:col-span-2 space-y-6">

                    <!-- 输入 SMILES -->
                    <div class="bg-white rounded-lg    border border-slate-200">
                        <div class="bg-gray-200 px-4 py-3    flex items-center">
                            <i class="fas fa-keyboard text-gray-600 mr-2"></i>
                            <span class="font-medium text-gray-700">By inputting SMILES</span>
                        </div>
                        <div class="p-4">
                            <div class="flex items-center space-x-4">
                                <label class="font-medium text-gray-700">SMILES：</label>
                                <input type="text" v-model="smilesInput"
                                    placeholder="CCN(CC)CCCOC1=NC2=C(C=CC(=C2Cl)C(=O)OC)C(=N1)C3=CC=CC=C3S(=O)(=O)O"
                                    class="flex-1 px-3 py-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500">
                                <button
                                    @click="loadExample('amlodipine')"
                                    class="bg-green-600 text-white px-3 py-1.5 text-sm rounded hover:bg-green-700 transition duration-200">
                                   eg: amlodipine besylate
                                </button>
                            </div>
                        </div>
                    </div>

                    <!-- 上传文件 -->
                    <div class="bg-white rounded-lg    border border-slate-200">
                        <div class="bg-gray-200 px-4 py-3    flex items-center justify-between">
                            <div class="flex items-center">
                                <i class="fas fa-upload text-gray-600 mr-2"></i>
                                <span class="font-medium text-gray-700">By Uploading Files (*.sdf)</span>
                            </div>
                            <button class="text-blue-600 hover:text-blue-800 text-sm flex items-center">
                                <i class="fas fa-exchange-alt mr-1"></i>
                                Format converter
                            </button>
                        </div>
                        <div class="p-4">
                            <div class="flex items-center space-x-4">
                                <label class="font-medium text-gray-700">Choose：</label>
                                <input 
                                    ref="fileInput"
                                    type="file" 
                                    accept=".sdf"
                                    @change="handleFileUpload"
                                    class="flex-1 px-3 py-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500">
                                <button
                                    @click="loadExample('compounds')"
                                    class="bg-green-600 text-white px-3 py-1.5 text-sm rounded hover:bg-green-700 transition duration-200">
                                    20 compounds
                                </button>
                            </div>
                            <!-- 文件上传状态显示 -->
                            <div v-if="uploadedFile" class="mt-3 p-3 bg-blue-50 border border-blue-200 rounded">
                                <div class="flex items-center justify-between">
                                    <div class="flex items-center">
                                        <i class="fas fa-file-alt text-blue-600 mr-2"></i>
                                        <span class="text-sm text-blue-800">{{ uploadedFile.name }}</span>
                                        <span class="text-xs text-blue-600 ml-2">({{ formatFileSize(uploadedFile.size) }})</span>
                                    </div>
                                    <button 
                                        @click="clearFile"
                                        class="text-red-600 hover:text-red-800 text-sm">
                                        <i class="fas fa-times"></i>
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- 分子编辑器 -->
                    <div class="bg-white rounded-lg border border-slate-200">
                        <div class="bg-gray-200 px-4 py-3 flex items-center">
                            <i class="fas fa-draw-polygon text-gray-600 mr-2"></i>
                            <span class="font-medium text-gray-700">By Drawing Molecule from Editor Below</span>
                        </div>
                        <div class="p-4">
                            <MolecularEditor 
                                @smiles-generated="handleSmilesGenerated"
                            />
                        </div>
                    </div>

                    <!-- 选择数据源 -->
                    <div class="bg-white rounded-lg    border border-slate-200">
                        <div class="bg-gray-200 px-4 py-3    flex items-center">
                            <i class="fas fa-database text-gray-600 mr-2"></i>
                            <span class="font-medium text-gray-700">Select the Data Source</span>
                        </div>
                        <div class="p-4">
                            <select v-model="dataSource"
                                class="w-full px-3 py-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500">
                                <option value="smiles">By Inputting SMILES</option>
                                <option value="file">By Uploading File</option>
                                <option value="editor">By Drawing Molecule from the Editor</option>
                            </select>
                        </div>
                    </div>

                    <!-- 提交按钮 -->
                    <div class="flex space-x-4">
                        <button
                            @click="evaluateDruglikeness"
                            :disabled="isLoading"
                            :class="[
                                'px-8 py-3 rounded font-medium transition duration-200 flex items-center',
                                isLoading 
                                    ? 'bg-gray-400 text-gray-200 cursor-not-allowed' 
                                    : 'bg-green-600 text-white hover:bg-green-700'
                            ]">
                            <i :class="[
                                'mr-2',
                                isLoading ? 'fas fa-spinner fa-spin' : 'fas fa-paper-plane'
                            ]"></i>
                            {{ isLoading ? 'Evaluating...' : 'Submit' }}
                        </button>
                        <button
                            @click="resetForm"
                            :disabled="isLoading"
                            class="bg-gray-500 text-white px-8 py-3 rounded font-medium hover:bg-gray-600 transition duration-200 flex items-center disabled:bg-gray-400 disabled:cursor-not-allowed">
                            <i class="fas fa-redo mr-2"></i>
                            Reset
                        </button>
                    </div>




                </div>
            </div>
        </main>
    </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import MolecularEditor from '@/components/MolecularEditor.vue'
import { storeResultData } from '@/utils/storage.js'
import message from '@/utils/message.js'

const router = useRouter()

// 规则数据
const rules = ref([
    { id: 'lipinski', name: "Lipinski's rules" },
    { id: 'ghose', name: "Ghose's rules" },
    { id: 'oprea', name: "Oprea's rules" },
    { id: 'veber', name: "Veber's rules" },
    { id: 'varma', name: "Varma's rules" }
])

// 选中的规则
const selectedRule = ref('lipinski')

// 不同规则的模型性能数据
const rulePerformanceData = {
  lipinski: [
    { label: 'Rule name', value: "Lipinski's rules" },
    { label: 'Content', value: 'MW<=500;\nlogP<=5;\nHacc<=10;\nHdon<=5' },
    { label: 'Reference', value: 'Adv Drug Deliver Rev, 2001, 46(1-3):3-26.' }
  ],
  ghose: [
    { label: 'Rule name', value: "Ghose's rules" },
    { label: 'Content', value: '- 5.6< MclogP < -0.4\nmean: 2.52;\n160 < MW < 480\nmean: 357;\n40 < MR < 130\nmean: 97;\n20 < natoms < 70\nmean: 48' },
    { label: 'Reference', value: 'J Comb Chem, 1999, 1(1):55-68.' }
  ],
  oprea: [
    { label: 'Rule name', value: "Oprea's rules" },
    { label: 'Content', value: 'nrings>=3;\nrigidbonds>=18;\nnRotbond>6' },
    { label: 'Reference', value: 'J Comput Aid Mol Des, 2000, 14(3):251-64.' }
  ],
  veber: [
    { label: 'Rule name', value: "Veber's rules" },
    { label: 'Content', value: 'nRotbond<=10;\ntPSA<= 140 or\nHacc and Hdon<=12' },
    { label: 'Reference', value: 'J Med Chem, 2002, 45(12):2615-23.' }
  ],
  varma: [
    { label: 'Rule name', value: "Varma's rules" },
    { label: 'Content', value: 'MW<= 500;\nTPSA<=125;\n-5< logD < = 2;\nHacc+Hdon<=9;\nnRotbond<=12' },
    { label: 'Reference', value: 'J Med Chem, 2010, 53(3):1098-108.' }
  ]
}

// 计算属性：当前选中规则的性能数据
const modelPerformance = computed(() => {
  return rulePerformanceData[selectedRule.value] || rulePerformanceData.lipinski
})

// 表单数据
const smilesInput = ref('')
const dataSource = ref('smiles')
const uploadedFile = ref(null)
const fileInput = ref(null)

// API相关状态
const isLoading = ref(false)

// 规则ID到API规则名称的映射
const ruleMapping = {
  'lipinski': 'Lipinski',
  'ghose': 'Ghose',
  'oprea': 'Oprea',
  'veber': 'Veber',
  'varma': 'Varma'
}

// 验证用户输入
const validateInput = () => {
  // 检查是否选择了规则
  if (!selectedRule.value) {
    message.warning({
      title: '输入提示',
      message: '请选择一个药物相似性规则进行评估',
      duration: 4000
    })
    return false
  }
  
  // 根据数据源验证输入
  if (dataSource.value === 'smiles' || dataSource.value === 'editor') {
    const smiles = smilesInput.value.trim()
    if (!smiles) {
      message.warning({
        title: '输入提示',
        message: dataSource.value === 'smiles' ? '请输入有效的SMILES字符串' : '请使用分子编辑器绘制分子结构或输入SMILES字符串',
        duration: 4000
      })
      return false
    }
  } else if (dataSource.value === 'file') {
    if (!uploadedFile.value) {
      message.warning({
        title: '输入提示',
        message: '请选择一个SDF格式的文件进行上传',
        duration: 4000
      })
      return false
    }
  }
  
  return true
}

// API调用函数
const evaluateDruglikeness = async () => {
  try {
    isLoading.value = true
    
    // 验证用户输入
    if (!validateInput()) {
      return
    }
    
    // 根据数据源类型选择不同的API调用方式
    if (dataSource.value === 'file') {
      await evaluateFromFile()
    } else {
      await evaluateFromSmiles()
    }
    
  } catch (error) {
    console.error('API调用错误:', error)
    // 处理网络错误和API错误
    let errorMsg = ''
    if (error.name === 'TypeError' && error.message.includes('fetch')) {
      errorMsg = '无法连接到服务器，请检查服务器是否运行在 http://localhost:8000'
    } else if (error.message.includes('Invalid SMILES')) {
      errorMsg = '无效的SMILES字符串，请检查输入格式是否正确'
    } else {
      errorMsg = error.message || '网络请求失败，请检查服务器连接'
    }
    
    // 使用消息提示组件显示错误
    message.error({
      title: '错误',
      message: errorMsg,
      duration: 5000
    })
  } finally {
    isLoading.value = false
  }
}

// SMILES评估函数
const evaluateFromSmiles = async () => {
  // 获取SMILES输入
  let smiles = ''
  if (dataSource.value === 'smiles') {
    smiles = smilesInput.value.trim()
  } else if (dataSource.value === 'editor') {
    smiles = smilesInput.value.trim() // 从编辑器获取的SMILES已存储在smilesInput中
  }
  
  if (!smiles) {
    throw new Error('请输入有效的SMILES字符串')
  }
  
  // 构建请求体 - 根据API文档更新
  const requestBody = {
    smiles: smiles,
    rules: [ruleMapping[selectedRule.value]] // 使用正确的规则名称映射
  }
  
  // 发送API请求 - 使用正确的API路径
  const response = await fetch('http://8.152.194.158:16666/api/v1/druglikeness/evaluate', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(requestBody)
  })
  
  const result = await response.json()
  
  // 根据API文档，成功时code为0
  if (result.code === 0) {
    // 显示成功消息
    message.success({
      title: '评估成功',
      message: 'SMILES分子评估完成，正在跳转到结果页面...',
      duration: 2000
    })
    
    // 使用存储工具函数存储数据
    const resultKey = storeResultData({
      data: result.data,
      rule: ruleMapping[selectedRule.value],
      source: 'smiles'
    })
    
    // 延迟跳转，让用户看到成功消息
    setTimeout(() => {
      router.push({
        path: '/druglikeness-result',
        query: {
          resultId: resultKey
        }
      })
    }, 1000)
    
    console.log('评估成功，跳转到结果页面:', result.data)
  } else {
    throw new Error(result.message || '评估失败')
  }
}

// SDF文件评估函数
const evaluateFromFile = async () => {
  if (!uploadedFile.value) {
    throw new Error('请选择SDF文件')
  }
  
  // 构建FormData
  const formData = new FormData()
  formData.append('file', uploadedFile.value)
  
  // 构建查询参数
  const queryParams = new URLSearchParams({
    selected_rule: ruleMapping[selectedRule.value],
    isomeric_smiles: 'true',
    kekule_smiles: 'true',
    canonical: 'true'
  })
  
  // 发送API请求 - 使用SDF分析接口
  const response = await fetch(`http://8.152.194.158:16666/api/v1/sdf/analyze?${queryParams}`, {
    method: 'POST',
    body: formData
  })
  
  const result = await response.json()
  
  // 根据API文档，成功时code为0
  if (result.code === 0) {
    // 显示成功消息
    message.success({
      title: '评估成功',
      message: 'SDF文件评估完成，正在跳转到结果页面...',
      duration: 2000
    })
    
    // 使用存储工具函数存储数据
    const resultKey = storeResultData({
      data: result.data,
      rule: ruleMapping[selectedRule.value],
      source: 'file'
    })
    
    // 延迟跳转，让用户看到成功消息
    setTimeout(() => {
      router.push({
        path: '/druglikeness-result',
        query: {
          resultId: resultKey
        }
      })
    }, 1000)
    
    console.log('SDF文件评估成功，跳转到结果页面:', result.data)
  } else {
    throw new Error(result.message || 'SDF文件评估失败')
  }
}

// 重置函数
const resetForm = () => {
  smilesInput.value = ''
  dataSource.value = 'smiles'
  selectedRule.value = 'lipinski'
  uploadedFile.value = null
  if (fileInput.value) {
    fileInput.value.value = ''
  }
  
  // 显示重置成功消息
  message.info({
    title: '表单已重置',
    message: '所有输入内容已清空，可以重新开始',
    duration: 2000
  })
}

// 文件上传处理函数
const handleFileUpload = (event) => {
  const file = event.target.files[0]
  if (file) {
    // 验证文件类型
    if (!file.name.toLowerCase().endsWith('.sdf')) {
      message.error({
        title: '文件格式错误',
        message: '请选择SDF格式的文件',
        duration: 4000
      })
      event.target.value = ''
      return
    }
    
    // 验证文件大小 (限制为10MB)
    const maxSize = 10 * 1024 * 1024 // 10MB
    if (file.size > maxSize) {
      message.error({
        title: '文件大小超限',
        message: '文件大小不能超过10MB',
        duration: 4000
      })
      event.target.value = ''
      return
    }
    
    uploadedFile.value = file
    dataSource.value = 'file'
    
    // 显示成功消息
    message.success({
      title: '文件上传成功',
      message: `已选择文件: ${file.name} (${formatFileSize(file.size)})`,
      duration: 3000
    })
    
    console.log('文件上传成功:', file.name, formatFileSize(file.size))
  }
}

// 清除文件
const clearFile = () => {
  uploadedFile.value = null
  if (fileInput.value) {
    fileInput.value.value = ''
  }
  dataSource.value = 'smiles'
}

// 格式化文件大小
const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  // 下载示例SDF文件
  const downloadExampleSdfFile = async () => {
    try {
      // 动态导入SDF文件
      const sdfUrl = new URL('../assets/eg.sdf', import.meta.url).href
      const response = await fetch(sdfUrl)
      const blob = await response.blob()
      
      // 创建下载链接
      const link = document.createElement('a')
      link.href = URL.createObjectURL(blob)
      link.download = 'eg.sdf'
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      
      // 清理URL对象
      URL.revokeObjectURL(link.href)
      
      // 显示成功消息
      message.success({
        title: '下载成功',
        message: '示例SDF文件已开始下载',
        duration: 3000
      })
    } catch (error) {
      console.error('下载示例文件失败:', error)
      message.error({
        title: '下载失败',
        message: '下载示例文件失败，请稍后重试',
        duration: 4000
      })
    }
  }

// 加载示例数据
const loadExample = (type) => {
  if (type === 'amlodipine') {
    smilesInput.value = 'CCN(CC)CCCOC1=NC2=C(C=CC(=C2Cl)C(=O)OC)C(=N1)C3=CC=CC=C3S(=O)(=O)O'
    dataSource.value = 'smiles'
    // 清除文件上传
    uploadedFile.value = null
    if (fileInput.value) {
      fileInput.value.value = ''
    }
    
    // 显示成功消息
    message.success({
      title: '示例加载成功',
      message: '已加载氨氯地平贝西酸盐的SMILES结构',
      duration: 3000
    })
  } else if (type === 'compounds') {
    // 下载示例SDF文件
    downloadExampleSdfFile()
  }
}

// 分子编辑器元素
const elements = ref(['C', 'N', 'O', 'S', 'P', 'Cl', 'Br', 'I', 'X'])

// 获取元素颜色
const getElementColor = (element) => {
    const colors = {
        'C': 'text-black bg-gray-200',
        'N': 'text-blue-600 bg-blue-100',
        'O': 'text-red-600 bg-red-100',
        'S': 'text-yellow-600 bg-yellow-100',
        'P': 'text-purple-600 bg-purple-100',
        'Cl': 'text-green-600 bg-green-100',
        'Br': 'text-orange-600 bg-orange-100',
        'I': 'text-indigo-600 bg-indigo-100',
        'X': 'text-gray-600 bg-gray-200'
    }
    return colors[element] || 'text-gray-600 bg-gray-200'
}

// 分子编辑器事件处理
const handleSmilesGenerated = (smiles) => {
    smilesInput.value = smiles
    dataSource.value = 'editor'
    

    
    console.log('Generated SMILES from editor:', smiles)
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