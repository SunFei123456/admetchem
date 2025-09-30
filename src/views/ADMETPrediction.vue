<template>
    <div class="bg-gray-50 min-h-screen">
        <main class="max-w-7xl mx-auto px-4 py-6">
            <div class="grid lg:grid-cols-12 gap-6">

                <!-- 左侧：属性选择器 -->
                <div class="lg:col-span-3 space-y-4">
                    <div class="bg-white rounded-lg border border-slate-200">
                        <div class="bg-gray-200 px-4 py-3 flex items-center">
                            <i class="fas fa-hand-pointer text-gray-600 mr-2"></i>
                            <span class="font-medium text-gray-700">Click to select & calculate</span>
                        </div>
                        <div class="p-4 space-y-4">

                            <!-- 理化性质 -->
                            <div>
                                <div class="flex items-center mb-2">
                                    <i class="fas fa-flask text-blue-600 mr-2"></i>
                                    <h3 class="font-semibold text-gray-800">Physicochemical Property</h3>
                                </div>
                                <div class="ml-6 space-y-2">
                                    <div v-for="prop in physicochemicalProps" :key="prop.id" class="flex items-center">
                                        <input type="checkbox" :id="prop.id" v-model="prop.selected"
                                            @change="updateSelectedModel(prop)"
                                            class="mr-2 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded">
                                        <label :for="prop.id"
                                            class="text-blue-600 hover:text-blue-800 cursor-pointer text-sm">
                                            {{ prop.name }}
                                        </label>
                                    </div>
                                </div>
                            </div>

                            <!-- 吸收 -->
                            <div>
                                <div class="flex items-center mb-2">
                                    <i class="fas fa-arrow-down text-green-600 mr-2"></i>
                                    <h3 class="font-semibold text-gray-800">Absorption</h3>
                                </div>
                                <div class="ml-6 space-y-2">
                                    <div v-for="prop in absorptionProps" :key="prop.id" class="flex items-center">
                                        <input type="checkbox" :id="prop.id" v-model="prop.selected"
                                            @change="updateSelectedModel(prop)"
                                            class="mr-2 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded">
                                        <label :for="prop.id"
                                            class="text-blue-600 hover:text-blue-800 cursor-pointer text-sm">
                                            {{ prop.name }}
                                        </label>
                                    </div>
                                </div>
                            </div>

                            <!-- 分布 -->
                            <div>
                                <div class="flex items-center mb-2">
                                    <i class="fas fa-share-alt text-purple-600 mr-2"></i>
                                    <h3 class="font-semibold text-gray-800">Distribution</h3>
                                </div>
                                <div class="ml-6 space-y-2">
                                    <div v-for="prop in distributionProps" :key="prop.id" class="flex items-center">
                                        <input type="checkbox" :id="prop.id" v-model="prop.selected"
                                            @change="updateSelectedModel(prop)"
                                            class="mr-2 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded">
                                        <label :for="prop.id"
                                            class="text-blue-600 hover:text-blue-800 cursor-pointer text-sm">
                                            {{ prop.name }}
                                        </label>
                                    </div>
                                </div>
                            </div>

                            <!-- 代谢 -->
                            <div>
                                <div class="flex items-center mb-2">
                                    <i class="fas fa-cogs text-orange-600 mr-2"></i>
                                    <h3 class="font-semibold text-gray-800">Metabolism</h3>
                                </div>
                                <div class="ml-6 space-y-2">
                                    <div v-for="prop in metabolismProps" :key="prop.id" class="flex items-center">
                                        <input type="checkbox" :id="prop.id" v-model="prop.selected"
                                            @change="updateSelectedModel(prop)"
                                            class="mr-2 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded">
                                        <label :for="prop.id"
                                            class="text-blue-600 hover:text-blue-800 cursor-pointer text-sm">
                                            {{ prop.name }}
                                        </label>
                                    </div>
                                </div>
                            </div>

                            <!-- 排泄 -->
                            <div>
                                <div class="flex items-center mb-2">
                                    <i class="fas fa-arrow-up text-red-600 mr-2"></i>
                                    <h3 class="font-semibold text-gray-800">Excretion</h3>
                                </div>
                                <div class="ml-6 space-y-2">
                                    <div v-for="prop in excretionProps" :key="prop.id" class="flex items-center">
                                        <input type="checkbox" :id="prop.id" v-model="prop.selected"
                                            @change="updateSelectedModel(prop)"
                                            class="mr-2 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded">
                                        <label :for="prop.id"
                                            class="text-blue-600 hover:text-blue-800 cursor-pointer text-sm">
                                            {{ prop.name }}
                                        </label>
                                    </div>
                                </div>
                            </div>

                            <!-- 毒性 -->
                            <div>
                                <div class="flex items-center mb-2">
                                    <i class="fas fa-exclamation-triangle text-red-600 mr-2"></i>
                                    <h3 class="font-semibold text-gray-800">Toxicity</h3>
                                </div>
                                <div class="ml-6 space-y-2">
                                    <div v-for="prop in toxicityProps" :key="prop.id" class="flex items-center">
                                        <input type="checkbox" :id="prop.id" v-model="prop.selected"
                                            @change="updateSelectedModel(prop)"
                                            class="mr-2 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded">
                                        <label :for="prop.id"
                                            class="text-blue-600 hover:text-blue-800 cursor-pointer text-sm">
                                            {{ prop.name }}
                                        </label>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 中间：输入方式 -->
                <div class="lg:col-span-6 space-y-6">

                    <!-- 输入 SMILES -->
                    <div class="bg-white rounded-lg border border-slate-200">
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
                                <button
                                    class="bg-green-600 text-white px-2 py-1 text-[10px] rounded hover:bg-green-700 transition duration-200">
                                    Example: Omeprazole
                                </button>
                            </div>
                        </div>
                    </div>

                    <!-- 上传文件 -->
                    <div class="bg-white rounded-lg border border-slate-200">
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
                                <input type="file" accept=".sdf"
                                    class="flex-1 px-3 py-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500">
                                <button
                                    class="bg-green-600 text-white px-2 py-1 text-[10px] rounded hover:bg-green-700 transition duration-200">
                                    Example: 20 compounds
                                </button>
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
                                @structure-changed="handleStructureChanged"
                            />
                        </div>
                    </div>

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

                    <!-- 提交按钮 -->
                    <div class="flex space-x-4">
                        <button
                            class="bg-green-600 text-white px-8 py-3 rounded font-medium hover:bg-green-700 transition duration-200 flex items-center">
                            <i class="fas fa-paper-plane mr-2"></i>
                            Submit
                        </button>
                        <button
                            class="bg-gray-500 text-white px-8 py-3 rounded font-medium hover:bg-gray-600 transition duration-200 flex items-center">
                            <i class="fas fa-redo mr-2"></i>
                            Reset
                        </button>
                    </div>
                </div>

                <!-- 右侧：模型信息 -->
                <div class="lg:col-span-3">
                    <div class="bg-white rounded-lg border border-slate-200">
                        <div class="bg-gray-200 px-4 py-3 flex items-center">
                            <i class="fas fa-info-circle text-gray-600 mr-2"></i>
                            <span class="font-medium text-gray-700">Model information</span>
                        </div>
                        <div class="p-4">
                            <div v-if="selectedModel">
                                <h4 class="font-semibold text-gray-800 mb-3">{{ selectedModel.name }}</h4>
                                <div class="space-y-2 text-sm">
                                    <div class="flex justify-between">
                                        <span class="text-gray-600">R²:</span>
                                        <span class="font-medium">{{ selectedModel.r2 }}</span>
                                    </div>
                                    <div class="flex justify-between">
                                        <span class="text-gray-600">RMSE:</span>
                                        <span class="font-medium">{{ selectedModel.rmse }}</span>
                                    </div>
                                    <div class="flex justify-between">
                                        <span class="text-gray-600">MAE:</span>
                                        <span class="font-medium">{{ selectedModel.mae }}</span>
                                    </div>
                                </div>
                                <div class="mt-4 p-3 bg-blue-50 rounded">
                                    <p class="text-xs text-gray-600">{{ selectedModel.description }}</p>
                                </div>
                            </div>
                            <div v-else class="text-center text-gray-500">
                                <i class="fas fa-mouse-pointer text-2xl mb-2"></i>
                                <p class="text-sm">Please select a property to view model information</p>
                            </div>
                        </div>
                    </div>

                    <!-- 使用提示 -->
                    <div class="mt-6 bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                        <div class="flex items-start">
                            <i class="fas fa-lightbulb text-yellow-600 mr-2 mt-0.5"></i>
                            <div class="text-sm">
                                <h4 class="font-semibold text-base text-yellow-800 mb-1">Tips</h4>
                                <ul class="text-yellow-700 space-y-1 text-xs">
                                    <li>• Before calculation, users should prepare a valid SMILES or *.sdf file in case of errors</li>
                                    <li>• Users can use ChemSAR to generate valid SMILES or SDF files</li>
                                    <li>• Some invalid descriptor values (e.g., NaN, infinity or a value too large) will lead to errors</li>
                                    <li>• Select desired ADMET properties and submit for prediction results</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </main>
    </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import MolecularEditor from '@/components/MolecularEditor.vue'

// 理化性质
const physicochemicalProps = ref([
    { id: 'logS', name: 'logS', selected: false, r2: '0.82', rmse: '0.65', mae: '0.48', description: 'Aqueous solubility prediction using molecular descriptors' },
    { id: 'logD', name: 'logD7.4', selected: false, r2: '0.89', rmse: '0.52', mae: '0.41', description: 'Distribution coefficient at pH 7.4' },
    { id: 'logP', name: 'logP', selected: false, r2: '0.91', rmse: '0.45', mae: '0.35', description: 'Partition coefficient between octanol and water' }
])

// 吸收
const absorptionProps = ref([
    { id: 'caco2', name: 'Caco-2', selected: false, r2: '0.76', rmse: '0.58', mae: '0.42', description: 'Caco-2 cell permeability prediction' },
    { id: 'pgp', name: 'Pgp-inhibitor', selected: false, r2: '0.84', rmse: '0.31', mae: '0.24', description: 'P-glycoprotein inhibition prediction' },
    { id: 'hia', name: 'HIA', selected: false, r2: '0.88', rmse: '0.28', mae: '0.21', description: 'Human intestinal absorption prediction' },
    { id: 'bioavailability', name: 'Bioavailability', selected: false, r2: '0.79', rmse: '0.35', mae: '0.27', description: 'Oral bioavailability prediction' }
])

// 分布
const distributionProps = ref([
    { id: 'ppb', name: 'PPB', selected: false, r2: '0.81', rmse: '0.42', mae: '0.33', description: 'Plasma protein binding prediction' },
    { id: 'bbb', name: 'BBB', selected: false, r2: '0.85', rmse: '0.29', mae: '0.22', description: 'Blood-brain barrier penetration' },
    { id: 'vd', name: 'VD', selected: false, r2: '0.77', rmse: '0.51', mae: '0.39', description: 'Volume of distribution prediction' }
])

// 代谢
const metabolismProps = ref([
    { id: 'cyp1a2', name: 'CYP1A2-inhibitor', selected: false, r2: '0.83', rmse: '0.33', mae: '0.25', description: 'CYP1A2 enzyme inhibition prediction' },
    { id: 'cyp2c9', name: 'CYP2C9-inhibitor', selected: false, r2: '0.86', rmse: '0.31', mae: '0.23', description: 'CYP2C9 enzyme inhibition prediction' },
    { id: 'cyp2d6', name: 'CYP2D6-inhibitor', selected: false, r2: '0.84', rmse: '0.32', mae: '0.24', description: 'CYP2D6 enzyme inhibition prediction' }
])

// 排泄
const excretionProps = ref([
    { id: 'clearance', name: 'Clearance', selected: false, r2: '0.75', rmse: '0.48', mae: '0.36', description: 'Total clearance prediction' },
    { id: 't12', name: 'T1/2', selected: false, r2: '0.72', rmse: '0.52', mae: '0.41', description: 'Half-life prediction' }
])

// 毒性
const toxicityProps = ref([
    { id: 'ames', name: 'AMES', selected: false, r2: '0.87', rmse: '0.28', mae: '0.21', description: 'Mutagenicity prediction' },
    { id: 'carcinogenicity', name: 'Carcinogenicity', selected: false, r2: '0.82', rmse: '0.34', mae: '0.26', description: 'Carcinogenicity prediction' },
    { id: 'acute_toxicity', name: 'Acute Toxicity', selected: false, r2: '0.79', rmse: '0.41', mae: '0.32', description: 'Acute oral toxicity prediction' }
])

// 输入数据
const smilesInput = ref('')
const dataSource = ref('smiles')

// 选中的模型
const selectedModel = computed(() => {
    const allProps = [
        ...physicochemicalProps.value,
        ...absorptionProps.value,
        ...distributionProps.value,
        ...metabolismProps.value,
        ...excretionProps.value,
        ...toxicityProps.value
    ]
    return allProps.find(prop => prop.selected)
})

// 更新选中的模型
const updateSelectedModel = (prop) => {
    // 取消其他选中项
    const allProps = [
        ...physicochemicalProps.value,
        ...absorptionProps.value,
        ...distributionProps.value,
        ...metabolismProps.value,
        ...excretionProps.value,
        ...toxicityProps.value
    ]

    allProps.forEach(p => {
        if (p.id !== prop.id) {
            p.selected = false
        }
    })
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
</style>