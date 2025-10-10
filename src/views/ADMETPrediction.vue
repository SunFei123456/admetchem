<template>
    <div class="bg-gray-50 min-h-screen">
        <main class="max-w-7xl mx-auto px-4 py-6">
            <div class="grid lg:grid-cols-3 gap-6">

                <!-- 左侧：属性选择器 -->
                <div class="lg:col-span-1 space-y-6">
                    <div class="bg-white rounded-lg border border-slate-200">
                        <div class="bg-gray-200 px-4 py-3 flex items-center justify-between">
                            <div class="flex items-center">
                                <i class="fas fa-hand-pointer text-gray-600 mr-2"></i>
                                <span class="font-medium text-gray-700">Click to select & calculate</span>
                            </div>
                            <!-- 全选按钮 -->
                            <button @click="toggleSelectAll"
                                class="text-sm px-3 py-1 rounded transition-colors duration-200"
                                :class="isAllSelected ? 'bg-blue-600 text-white hover:bg-blue-700' : 'bg-gray-300 text-gray-700 hover:bg-gray-400'">
                                <i class="fas fa-check-double mr-1"></i>
                                {{ isAllSelected ? '取消全选' : '全选' }}
                            </button>
                        </div>
                        <div class="p-4 space-y-3">

                            <!-- Biophysics - 生物物理学性质 -->
                            <div class="border border-gray-200 rounded">
                                <div @click="toggleCategory('biophysics')"
                                    class="flex items-center justify-between p-3 cursor-pointer hover:bg-gray-50 transition-colors duration-200">
                                    <div class="flex items-center">
                                        <i class="fas fa-atom text-indigo-600 mr-2"></i>
                                        <h3 class="font-semibold text-gray-800">Biophysics</h3>
                                        <span class="ml-2 text-xs text-gray-500">({{ biophysicsProps.length }})</span>
                                    </div>
                                    <i :class="[
                                        'fas transition-transform duration-200',
                                        expandedCategories.biophysics ? 'fa-chevron-up' : 'fa-chevron-down'
                                    ]"></i>
                                </div>
                                <transition name="collapse">
                                    <div v-show="expandedCategories.biophysics" class="px-3 pb-3 space-y-2">
                                        <div v-for="prop in biophysicsProps" :key="prop.id" class="flex items-center">
                                            <input type="checkbox" :id="prop.id" v-model="selectedProperties"
                                                :value="prop.id" @change="updateSelectedModel(prop)"
                                                class="mr-3 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded">
                                            <label :for="prop.id" :class="[
                                                'cursor-pointer text-sm transition-colors duration-200',
                                                selectedProperties.has(prop.id)
                                                    ? 'text-blue-800 font-semibold'
                                                    : 'text-blue-600 hover:text-blue-800'
                                            ]">
                                                {{ prop.name }}
                                            </label>
                                        </div>
                                    </div>
                                </transition>
                            </div>

                            <!-- Physical Chemistry - 物理化学性质 -->
                            <div class="border border-gray-200 rounded">
                                <div @click="toggleCategory('physicalChemistry')"
                                    class="flex items-center justify-between p-3 cursor-pointer hover:bg-gray-50 transition-colors duration-200">
                                    <div class="flex items-center">
                                        <i class="fas fa-flask text-cyan-600 mr-2"></i>
                                        <h3 class="font-semibold text-gray-800">Physical Chemistry</h3>
                                        <span class="ml-2 text-xs text-gray-500">({{ physicalChemistryProps.length
                                        }})</span>
                                    </div>
                                    <i :class="[
                                        'fas transition-transform duration-200',
                                        expandedCategories.physicalChemistry ? 'fa-chevron-up' : 'fa-chevron-down'
                                    ]"></i>
                                </div>
                                <transition name="collapse">
                                    <div v-show="expandedCategories.physicalChemistry" class="px-3 pb-3 space-y-2">
                                        <div v-for="prop in physicalChemistryProps" :key="prop.id"
                                            class="flex items-center">
                                            <input type="checkbox" :id="prop.id" v-model="selectedProperties"
                                                :value="prop.id" @change="updateSelectedModel(prop)"
                                                class="mr-3 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded">
                                            <label :for="prop.id" :class="[
                                                'cursor-pointer text-sm transition-colors duration-200',
                                                selectedProperties.has(prop.id)
                                                    ? 'text-blue-800 font-semibold'
                                                    : 'text-blue-600 hover:text-blue-800'
                                            ]">
                                                {{ prop.name }}
                                            </label>
                                        </div>
                                    </div>
                                </transition>
                            </div>

                            <!-- Physiology - 生理学性质 -->
                            <div class="border border-gray-200 rounded">
                                <div @click="toggleCategory('physiology')"
                                    class="flex items-center justify-between p-3 cursor-pointer hover:bg-gray-50 transition-colors duration-200">
                                    <div class="flex items-center">
                                        <i class="fas fa-heartbeat text-pink-600 mr-2"></i>
                                        <h3 class="font-semibold text-gray-800">Physiology</h3>
                                        <span class="ml-2 text-xs text-gray-500">({{ physiologyProps.length }})</span>
                                    </div>
                                    <i :class="[
                                        'fas transition-transform duration-200',
                                        expandedCategories.physiology ? 'fa-chevron-up' : 'fa-chevron-down'
                                    ]"></i>
                                </div>
                                <transition name="collapse">
                                    <div v-show="expandedCategories.physiology" class="px-3 pb-3 space-y-2">
                                        <div v-for="prop in physiologyProps" :key="prop.id" class="flex items-center">
                                            <input type="checkbox" :id="prop.id" v-model="selectedProperties"
                                                :value="prop.id" @change="updateSelectedModel(prop)"
                                                class="mr-3 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded">
                                            <label :for="prop.id" :class="[
                                                'cursor-pointer text-sm transition-colors duration-200',
                                                selectedProperties.has(prop.id)
                                                    ? 'text-blue-800 font-semibold'
                                                    : 'text-blue-600 hover:text-blue-800'
                                            ]">
                                                {{ prop.name }}
                                            </label>
                                        </div>
                                    </div>
                                </transition>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 右侧：输入方式 -->
                <div class="lg:col-span-2 space-y-6">

                    <!-- 选择数据源 -->
                    <div class="bg-white rounded-lg border border-slate-200">
                        <div class="bg-gray-200 px-4 py-3 flex items-center">
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

                    <!-- 输入 SMILES - 仅当选择 SMILES 时显示 -->
                    <div v-if="dataSource === 'smiles'" class="bg-white rounded-lg border border-slate-200">
                        <div class="bg-gray-200 px-4 py-3 flex items-center">
                            <i class="fas fa-keyboard text-gray-600 mr-2"></i>
                            <span class="font-medium text-gray-700">By inputting SMILES</span>
                        </div>
                        <div class="p-4">
                            <div class="flex items-center space-x-4">
                                <label class="font-medium text-gray-700">SMILES：</label>
                                <input type="text" v-model="smilesInput"
                                    placeholder="CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=C(N2)C=C(C=C3)OC"
                                    class="flex-1 px-3 py-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500">
                                <button @click="loadExample"
                                    class="bg-green-600 text-white px-3 py-1.5 text-sm rounded hover:bg-green-700 transition duration-200">
                                    <i class="fas fa-flask mr-1"></i>
                                    eg: Omeprazole
                                </button>
                            </div>
                        </div>
                    </div>

                    <!-- 上传文件 - 仅当选择文件时显示 -->
                    <div v-if="dataSource === 'file'" class="bg-white rounded-lg border border-slate-200">
                        <div class="bg-gray-200 px-4 py-3 flex items-center justify-between">
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
                                <input type="file" accept=".sdf" @change="handleFileChange"
                                    class="flex-1 px-3 py-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500">
                                <button v-if="uploadedFile"
                                    class="bg-green-600 text-white px-3 py-1.5 text-sm rounded hover:bg-green-700 transition duration-200">
                                    <i class="fas fa-file mr-1"></i>
                                    {{ uploadedFile.name }}
                                </button>
                            </div>
                        </div>
                    </div>

                    <!-- 分子编辑器 - 仅当选择编辑器时显示 -->
                    <div v-if="dataSource === 'editor'" class="bg-white rounded-lg border border-slate-200">
                        <div class="bg-gray-200 px-4 py-3 flex items-center">
                            <i class="fas fa-draw-polygon text-gray-600 mr-2"></i>
                            <span class="font-medium text-gray-700">By Drawing Molecule from Editor Below</span>
                        </div>
                        <div class="p-4">
                            <MolecularEditor @smiles-generated="handleSmilesGenerated"
                                @structure-changed="handleStructureChanged" />
                        </div>
                    </div>

                    <!-- 模型选择 -->
                    <div class="bg-white rounded-lg border border-slate-200">
                        <div class="bg-gray-200 px-4 py-3 flex items-center">
                            <i class="fas fa-brain text-gray-600 mr-2"></i>
                            <span class="font-medium text-gray-700">Select Model</span>
                        </div>
                        <div class="p-4">
                            <div class="flex items-center space-x-6">
                                <label class="flex items-center cursor-pointer group">
                                    <input type="radio" name="model" value="model1" v-model="selectedModel"
                                        class="mr-2 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300">
                                    <span :class="[
                                        'text-sm transition-colors duration-200',
                                        selectedModel === 'model1'
                                            ? 'text-blue-600'
                                            : 'text-gray-700 group-hover:text-blue-600'
                                    ]">
                                        <i class="fas fa-cube mr-1"></i>model-1
                                    </span>
                                </label>
                                <label class="flex items-center cursor-pointer group">
                                    <input type="radio" name="model" value="model2" v-model="selectedModel"
                                        class="mr-2 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300">
                                    <span :class="[
                                        'text-sm transition-colors duration-200',
                                        selectedModel === 'model2'
                                            ? 'text-blue-600'
                                            : 'text-gray-700 group-hover:text-blue-600'
                                    ]">
                                        <i class="fas fa-cubes mr-1"></i>model-2
                                    </span>
                                </label>
                            </div>
                        </div>
                    </div>

                    <!-- 提交按钮 -->
                    <div class="flex space-x-4">
                        <button @click="handleSubmit" :disabled="selectedPropsArray.length === 0 || isLoading"
                            class="bg-green-600 text-white px-8 py-3 rounded font-medium hover:bg-green-700 transition duration-200 flex items-center disabled:bg-gray-400 disabled:cursor-not-allowed">
                            <i :class="[isLoading ? 'fas fa-spinner fa-spin' : 'fas fa-paper-plane', 'mr-2']"></i>
                            {{ isLoading ? '预测中...' : `Submit (${selectedPropsArray.length} selected)` }}
                        </button>
                        <button @click="handleReset" :disabled="isLoading"
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
import { predictADMET, analyzeSdfFileADMET } from '@/api'
import message from '@/utils/message'

// Biophysics - 生物物理学性质
const biophysicsProps = ref([
    { id: 'cyp1a2', name: 'CYP1A2-inh', selected: false },
    { id: 'cyp2c9', name: 'CYP2C9-inh', selected: false },
    { id: 'cyp2c9-sub', name: 'CYP2C9-sub', selected: false },
    { id: 'cyp2c19', name: 'CYP2C19-inh', selected: false },
    { id: 'cyp2d6', name: 'CYP2D6-inh', selected: false },
    { id: 'cyp2d6-sub', name: 'CYP2D6-sub', selected: false },
    { id: 'cyp3a4', name: 'CYP3A4-inh', selected: false },
    { id: 'cyp3a4-sub', name: 'CYP3A4-sub', selected: false },
    { id: 'herg', name: 'hERG blockers', selected: false },
    { id: 'pgp', name: 'Pgp-inh', selected: false }
])

// Physical Chemistry - 物理化学性质
const physicalChemistryProps = ref([
    { id: 'logS', name: 'LogS', selected: false },
    { id: 'logP', name: 'LogP', selected: false },
    { id: 'logD', name: 'LogD', selected: false },
    { id: 'hydration', name: 'Hydration Free Energy', selected: false },
    { id: 'pampa', name: 'PAMPA', selected: false }
])

// Physiology - 生理学性质
const physiologyProps = ref([
    { id: 'ames', name: 'Ames', selected: false },
    { id: 'bbb', name: 'BBB', selected: false },
    { id: 'bioavailability', name: 'Bioavailability', selected: false },
    { id: 'caco2', name: 'Caco-2', selected: false },
    { id: 'clearance', name: 'CL', selected: false },
    { id: 'dili', name: 'DILI', selected: false },
    { id: 'halflife', name: 'Drug Half-Life', selected: false },
    { id: 'hia', name: 'HIA', selected: false },
    // { id: 'ld50', name: 'LD50', selected: false },  // 暂不支持，模型未提供
    { id: 'ppbr', name: 'PPBR', selected: false },
    { id: 'skinSen', name: 'SkinSen', selected: false },
    { id: 'nr-ar-lbd', name: 'NR-AR-LBD', selected: false },
    { id: 'nr-ar', name: 'NR-AR', selected: false },
    { id: 'nr-ahr', name: 'NR-AhR', selected: false },
    { id: 'nr-aromatase', name: 'NR-Aromatase', selected: false },
    { id: 'nr-er', name: 'NR-ER', selected: false },
    { id: 'nr-er-lbd', name: 'NR-ER-LBD', selected: false },
    { id: 'nr-ppar-gamma', name: 'NR-PPAR-gamma', selected: false },
    { id: 'sr-are', name: 'SR-ARE', selected: false },
    { id: 'sr-atad5', name: 'SR-ATAD5', selected: false },
    { id: 'sr-hse', name: 'SR-HSE', selected: false },
    { id: 'sr-mmp', name: 'SR-MMP', selected: false },
    { id: 'sr-p53', name: 'SR-p53', selected: false },
    { id: 'vdss', name: 'VDss', selected: false }
])

// 输入数据
const smilesInput = ref('')
const dataSource = ref('smiles')
const uploadedFile = ref(null)
const isLoading = ref(false)
const selectedModel = ref('model1')  // 默认选择模型1

// 路由
const router = useRouter()

// 选中的属性集合（多选）
const selectedProperties = ref(new Set())

// 折叠状态管理
const expandedCategories = ref({
    biophysics: true,
    physicalChemistry: false,
    physiology: false
})

// 切换分类折叠状态
const toggleCategory = (category) => {
    expandedCategories.value[category] = !expandedCategories.value[category]
}

// 计算属性：是否全选
const isAllSelected = computed(() => {
    const allProps = [
        ...biophysicsProps.value,
        ...physicalChemistryProps.value,
        ...physiologyProps.value
    ]
    return selectedProperties.value.size === allProps.length && allProps.length > 0
})

// 全选/取消全选功能
const toggleSelectAll = () => {
    if (isAllSelected.value) {
        // 取消全选
        selectedProperties.value.clear()
    } else {
        // 全选
        const allProps = [
            ...biophysicsProps.value,
            ...physicalChemistryProps.value,
            ...physiologyProps.value
        ]
        allProps.forEach(prop => {
            selectedProperties.value.add(prop.id)
        })
    }
}

// 更新选中的模型（多选逻辑）
// 注意：由于使用了v-model，实际上不需要这个函数
// 但保留以防需要额外的逻辑
const updateSelectedModel = (prop) => {
    // Set 的 v-model 会自动处理选中状态，这里不需要额外逻辑
    console.log('Selected properties:', selectedPropsArray.value)
}

// 获取选中的属性数组
const selectedPropsArray = computed(() => {
    return Array.from(selectedProperties.value)
})

// 加载示例SMILES
const loadExample = () => {
    // Omeprazole 的 SMILES
    smilesInput.value = 'CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=C(N2)C=C(C=C3)OC'
    message.success('已加载示例分子：Omeprazole')
}

// 文件上传处理
const handleFileChange = (event) => {
    const file = event.target.files[0]
    if (file) {
        uploadedFile.value = file
        console.log('File uploaded:', file.name)
    }
}

// 分子编辑器事件处理
const handleSmilesGenerated = (smiles) => {
    smilesInput.value = smiles
    dataSource.value = 'editor'
    console.log('Generated SMILES from editor:', smiles)
}

const handleStructureChanged = (structure) => {
    console.log('Structure changed:', structure)
}

// 表单验证
const validateInput = () => {
    // 检查是否选择了至少一个属性
    if (selectedProperties.value.size === 0) {
        message.warning('请至少选择一个ADMET属性')
        return false
    }

    // 根据数据源类型验证输入
    if (dataSource.value === 'smiles' || dataSource.value === 'editor') {
        if (!smilesInput.value || smilesInput.value.trim() === '') {
            message.warning('请输入SMILES字符串')
            return false
        }
    } else if (dataSource.value === 'file') {
        if (!uploadedFile.value) {
            message.warning('请上传SDF文件')
            return false
        }
    }

    return true
}

// 提交预测请求
const handleSubmit = async () => {
    if (!validateInput()) {
        return
    }

    isLoading.value = true

    try {
        const propertyIds = Array.from(selectedProperties.value)

        let response

        if (dataSource.value === 'file') {
            // SDF文件上传
            const formData = new FormData()
            formData.append('file', uploadedFile.value)
            formData.append('property_ids', propertyIds.join(','))

            response = await analyzeSdfFileADMET(formData)
        } else {
            // SMILES预测（包括手动输入和编辑器生成）
            response = await predictADMET({
                smiles: smilesInput.value.trim(),
                property_ids: propertyIds
            })
        }

        // 处理响应
        if (response.code === 0) {
            message.success('ADMET预测成功')

            // 跳转到结果页面，并传递结果数据
            router.push({
                name: 'admet-prediction-result',
                state: {
                    result: response.data,
                    dataSource: dataSource.value,
                    selectedProperties: propertyIds
                }
            })
        } else {
            message.error(response.message || 'ADMET预测失败')
        }
    } catch (error) {
        console.error('ADMET prediction error:', error)
        message.error(error.message || 'ADMET预测失败，请检查输入或稍后重试')
    } finally {
        isLoading.value = false
    }
}

// 重置表单
const handleReset = () => {
    smilesInput.value = ''
    uploadedFile.value = null
    selectedProperties.value.clear()
    dataSource.value = 'smiles'
    selectedModel.value = 'model1'  // 重置为默认模型

    message.info('表单已重置')
}
</script>

<style scoped>
/* 自定义样式 */
.transition-all {
    transition: all 0.3s ease;
}

/* 输入框聚焦效果 */
input:focus,
select:focus {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

/* 按钮悬停效果 */
button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

/* 复选框样式优化 */
input[type="checkbox"]:checked {
    background-color: #3b82f6;
    border-color: #3b82f6;
}

/* 折叠动画 */
.collapse-enter-active,
.collapse-leave-active {
    transition: all 0.3s ease;
    max-height: 1000px;
    overflow: hidden;
}

.collapse-enter-from,
.collapse-leave-to {
    max-height: 0;
    opacity: 0;
}
</style>