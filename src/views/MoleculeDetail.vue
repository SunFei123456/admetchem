<template>
    <div class="bg-gray-50 min-h-screen">
        <main class="max-w-7xl mx-auto px-4 py-6">
            <!-- 返回按钮 -->
            <div class="mb-6">
                <button @click="goBack"
                    class="flex items-center text-blue-600 hover:text-blue-800 transition duration-200">
                    <i class="fas fa-arrow-left mr-2"></i>
                    Back to Results
                </button>
            </div>

            <!-- 分子结构展示 -->
            <div class="grid grid-cols-3 gap-6 mb-6">
                <!-- 2D图 -->
                <div class="bg-white rounded-lg border border-slate-200 p-6">
                    <h3 class="text-lg font-bold text-gray-800 mb-4 text-center">2D Structure</h3>
                    <div id="molecule-2d" class="flex items-center justify-center bg-gray-50 rounded-lg"
                        style="min-height: 300px;">
                        <div v-if="!molecule" class="text-gray-400">Loading...</div>
                    </div>
                </div>

                <!-- 3D图 -->
                <div class="bg-white rounded-lg border border-slate-200 p-6">
                    <h3 class="text-lg font-bold text-gray-800 mb-4 text-center">3D Structure</h3>
                    <div id="molecule-3d" class="flex items-center justify-center bg-gray-50 rounded-lg"
                        style="min-height: 300px; width: 100%; height: 300px;">
                        <div v-if="!molecule" class="text-gray-400">Loading...</div>
                    </div>
                    <p class="text-xs text-center text-gray-500 mt-2">
                        <i class="fas fa-mouse-pointer mr-1"></i>Drag to rotate • Scroll to zoom
                    </p>
                </div>

                <!-- 雷达图 -->
                <div class="bg-white rounded-lg border border-slate-200 p-6">
                    <h3 class="text-lg font-bold text-gray-800 mb-4 text-center">Property Radar</h3>
                    <div id="radar-chart" class="flex items-center justify-center bg-gray-50 rounded-lg"
                        style="min-height: 300px;">
                        <div class="text-gray-400">Radar Chart</div>
                    </div>
                </div>
            </div>

            <!-- 分子详细信息 -->
            <div class="bg-white rounded-lg border border-slate-200 mb-6">
                <div class="bg-blue-600 text-white px-4 py-3 flex items-center">
                    <i class="fas fa-info-circle mr-2"></i>
                    <span class="font-medium">Molecule Information</span>
                </div>
                <div class="p-6">
                    <div class="grid grid-cols-1 gap-4">
                        <div class="flex items-start border-b pb-3">
                            <span class="font-semibold text-gray-700 w-32">SMILES:</span>
                            <span class="font-mono text-sm text-gray-900 break-all flex-1">{{ molecule?.smiles || 'N/A'
                                }}</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- 属性分组展示 -->
            <div class="grid grid-cols-3 gap-6">
                <!-- Biophysics -->
                <div class="bg-white rounded-lg border border-slate-200">
                    <div class="bg-purple-100 px-4 py-3 border-b">
                        <h3 class="font-bold text-gray-800">Biophysics</h3>
                    </div>
                    <div class="p-4">
                        <div v-for="prop in biophysicsProps" :key="prop.key" class="mb-3 last:mb-0">
                            <div class="flex items-center justify-between">
                                <span class="text-sm font-medium text-gray-700">{{ prop.label }}</span>
                                <div class="flex items-center">
                                    <!-- 符号显示 -->
                                    <template v-if="prop.symbol">
                                        <div v-if="prop.symbol.includes('-')"
                                            class="w-4 h-4 rounded-full bg-red-500 shadow-md"></div>
                                        <div v-else-if="prop.symbol.includes('+')"
                                            class="w-4 h-4 rounded-full bg-green-500 shadow-md"></div>
                                    </template>
                                    <!-- 数值显示 -->
                                    <template v-else>
                                        <span class="text-sm font-semibold text-gray-900">{{ prop.displayValue }}</span>
                                    </template>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Physical Chemistry -->
                <div class="bg-white rounded-lg border border-slate-200">
                    <div class="bg-blue-100 px-4 py-3 border-b">
                        <h3 class="font-bold text-gray-800">Physical Chemistry</h3>
                    </div>
                    <div class="p-4">
                        <div v-for="prop in physicalChemistryProps" :key="prop.key" class="mb-3 last:mb-0">
                            <div class="flex items-center justify-between">
                                <span class="text-sm font-medium text-gray-700">{{ prop.label }}</span>
                                <div class="flex items-center">
                                    <!-- 符号显示 -->
                                    <template v-if="prop.symbol">
                                        <div v-if="prop.symbol.includes('-')"
                                            class="w-4 h-4 rounded-full bg-red-500 shadow-md"></div>
                                        <div v-else-if="prop.symbol.includes('+')"
                                            class="w-4 h-4 rounded-full bg-green-500 shadow-md"></div>
                                    </template>
                                    <!-- 数值显示 -->
                                    <template v-else>
                                        <span class="text-sm font-semibold text-gray-900">{{ prop.displayValue }}</span>
                                    </template>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Physiology -->
                <div class="bg-white rounded-lg border border-slate-200">
                    <div class="bg-green-100 px-4 py-3 border-b">
                        <h3 class="font-bold text-gray-800">Physiology</h3>
                    </div>
                    <div class="p-4">
                        <div v-for="prop in physiologyProps" :key="prop.key" class="mb-3 last:mb-0">
                            <div class="flex items-center justify-between">
                                <span class="text-sm font-medium text-gray-700">{{ prop.label }}</span>
                                <div class="flex items-center">
                                    <!-- 符号显示 -->
                                    <template v-if="prop.symbol">
                                        <div v-if="prop.symbol.includes('-')"
                                            class="w-4 h-4 rounded-full bg-red-500 shadow-md"></div>
                                        <div v-else-if="prop.symbol.includes('+')"
                                            class="w-4 h-4 rounded-full bg-green-500 shadow-md"></div>
                                    </template>
                                    <!-- 数值显示 -->
                                    <template v-else>
                                        <span class="text-sm font-semibold text-gray-900">{{ prop.displayValue }}</span>
                                    </template>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </main>
    </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()
const molecule = ref(null)
let viewer3d = null

// 访问全局的 $3Dmol 对象
const $3Dmol = window.$3Dmol

// 属性分类映射
const propertyCategories = {
    biophysics: ['cyp1a2', 'cyp2c9', 'cyp2c9-sub', 'cyp2c19', 'cyp2d6', 'cyp2d6-sub', 'cyp3a4', 'cyp3a4-sub', 'herg', 'pgp'],
    physicalChemistry: ['logS', 'logP', 'logD', 'hydration', 'pampa'],
    physiology: ['ames', 'bbb', 'bioavailability', 'caco2', 'clearance', 'dili', 'halflife', 'hia', 'ppbr', 'skinSen',
        'nr-ar-lbd', 'nr-ar', 'nr-ahr', 'nr-aromatase', 'nr-er', 'nr-er-lbd', 'nr-ppar-gamma',
        'sr-are', 'sr-atad5', 'sr-hse', 'sr-mmp', 'sr-p53', 'vdss']
}

// 计算各类别的属性
const biophysicsProps = computed(() => {
    return getPropertiesByCategory('biophysics')
})

const physicalChemistryProps = computed(() => {
    return getPropertiesByCategory('physicalChemistry')
})

const physiologyProps = computed(() => {
    return getPropertiesByCategory('physiology')
})

// 根据分类获取属性
const getPropertiesByCategory = (category) => {
    if (!molecule.value?.predictions) return []

    const categoryKeys = propertyCategories[category]
    const props = []

    categoryKeys.forEach(key => {
        const prediction = molecule.value.predictions[key]
        if (prediction) {
            let displayValue = 'N/A'
            let symbol = null

            if (prediction.symbol) {
                symbol = prediction.symbol
            } else if (prediction.probability != null) {
                displayValue = prediction.probability.toFixed(2)
            } else if (prediction.value != null) {
                displayValue = prediction.value.toFixed(2)
            }

            props.push({
                key,
                label: prediction.name || key,
                symbol,
                displayValue
            })
        }
    })

    return props
}

// 页面初始化
onMounted(async () => {
    // 从 sessionStorage 中获取分子数据
    const moleculeDataStr = sessionStorage.getItem('currentMolecule')

    if (moleculeDataStr) {
        try {
            molecule.value = JSON.parse(moleculeDataStr)

            // 延迟加载分子结构，确保DOM已渲染
            setTimeout(async () => {
                await render2DStructure()
                await render3DStructure()
            }, 100)
        } catch (error) {
            console.error('Error parsing molecule data:', error)
            router.back()
        }
    } else {
        // 如果没有数据，返回上一页
        router.back()
    }
})

// 渲染2D结构
const render2DStructure = async () => {
    const container = document.getElementById('molecule-2d')
    if (!container || !molecule.value) return

    try {
        // 方法1: 尝试使用 RDKit.js
        if (window.RDKit) {
            try {
                await window.RDKit
                const rdkit = window.RDKit
                const mol = rdkit.get_mol(molecule.value.smiles)

                if (mol && mol.is_valid()) {
                    const svg = mol.get_svg()
                    mol.delete()

                    container.innerHTML = `
                        <div class="text-center">
                            <div class="inline-block bg-white p-4 rounded-lg border border-gray-200">
                                ${svg}
                            </div>
                            <p class="text-xs text-gray-400 mt-3">2D Structure (RDKit)</p>
                        </div>
                    `
                    return
                } else {
                    if (mol) mol.delete()
                }
            } catch (error) {
                console.error('RDKit rendering failed:', error)
            }
        }

        // 方法2: 使用 PubChem API 获取 2D 图像
        try {
            const smiles = encodeURIComponent(molecule.value.smiles)
            const imageUrl = `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/${smiles}/PNG`

            // 测试图像是否可加载
            const img = new Image()
            img.onload = () => {
                container.innerHTML = `
                    <div class="text-center">
                        <div class="inline-block bg-white p-4 rounded-lg border border-gray-200">
                            <img src="${imageUrl}" alt="2D Structure" style="max-width: 250px; max-height: 250px;" />
                        </div>
                        <p class="text-xs text-gray-400 mt-3">2D Structure (PubChem)</p>
                    </div>
                `
            }
            img.onerror = () => {
                // 方法3: 显示 SMILES 文本作为降级方案
                container.innerHTML = `
                    <div class="text-center p-8">
                        <div class="text-gray-600 mb-2">
                            <i class="fas fa-flask text-4xl text-blue-500"></i>
                        </div>
                        <p class="text-sm text-gray-500 mb-3">2D Structure</p>
                        <div class="bg-white p-4 rounded-lg border border-gray-200 inline-block max-w-xs">
                            <p class="text-xs text-gray-700 font-mono break-all">${molecule.value.smiles}</p>
                        </div>
                        <p class="text-xs text-gray-400 mt-3">SMILES Notation</p>
                    </div>
                `
            }
            img.src = imageUrl
        } catch (error) {
            console.error('PubChem API failed:', error)
            // 降级方案：显示 SMILES
            container.innerHTML = `
                <div class="text-center p-8">
                    <div class="text-gray-600 mb-2">
                        <i class="fas fa-flask text-4xl text-blue-500"></i>
                    </div>
                    <p class="text-sm text-gray-500 mb-3">2D Structure</p>
                    <div class="bg-white p-4 rounded-lg border border-gray-200 inline-block max-w-xs">
                        <p class="text-xs text-gray-700 font-mono break-all">${molecule.value.smiles}</p>
                    </div>
                    <p class="text-xs text-gray-400 mt-3">SMILES Notation</p>
                </div>
            `
        }
    } catch (error) {
        console.error('Error rendering 2D structure:', error)
        container.innerHTML = `
            <div class="text-center p-8">
                <div class="text-red-600 mb-2">
                    <i class="fas fa-exclamation-triangle text-4xl"></i>
                </div>
                <p class="text-sm text-gray-500">Error loading 2D structure</p>
            </div>
        `
    }
}

// 渲染3D结构
const render3DStructure = async () => {
    const container = document.getElementById('molecule-3d')
    if (!container || !molecule.value) return

    try {
        // 检查 $3Dmol 是否已加载
        if (!window.$3Dmol) {
            console.error('3Dmol.js library not loaded')
            container.innerHTML = `
                <div class="text-center p-8">
                    <div class="text-amber-600 mb-2">
                        <i class="fas fa-exclamation-circle text-4xl"></i>
                    </div>
                    <p class="text-sm text-gray-500 mb-2">3D Viewer Loading...</p>
                    <p class="text-xs text-gray-400">Please refresh the page if this persists</p>
                </div>
            `
            return
        }

        // 清空容器
        container.innerHTML = ''

        // 创建3Dmol查看器
        viewer3d = window.$3Dmol.createViewer(container, {
            backgroundColor: 'white',
        })

        // 从SMILES生成3D结构
        // 注意：3Dmol.js 不能直接从 SMILES 生成坐标，通常需要先转换为 SDF/MOL/PDB 格式
        // 这里我们使用一个在线服务或显示提示
        try {
            // 尝试使用 NIH/NCI CACTUS 服务将 SMILES 转换为 SDF
            const smiles = encodeURIComponent(molecule.value.smiles)
            const response = await fetch(`https://cactus.nci.nih.gov/chemical/structure/${smiles}/sdf`)

            if (response.ok) {
                const sdfData = await response.text()
                viewer3d.addModel(sdfData, 'sdf')
                viewer3d.setStyle({}, { stick: { radius: 0.15 }, sphere: { scale: 0.3 } })
                viewer3d.zoomTo()
                viewer3d.render()
                viewer3d.zoom(1.2, 1000)
            } else {
                throw new Error('Failed to fetch 3D structure')
            }
        } catch (error) {
            console.error('Error loading 3D structure:', error)
            // 显示错误提示
            container.innerHTML = `
                <div class="text-center p-8">
                    <div class="text-gray-600 mb-2">
                        <i class="fas fa-cube text-4xl text-amber-500"></i>
                    </div>
                    <p class="text-sm text-gray-500 mb-2">3D Structure</p>
                    <p class="text-xs text-gray-400">Unable to generate 3D coordinates</p>
                    <p class="text-xs text-gray-400 mt-1">(External service unavailable)</p>
                </div>
            `
        }
    } catch (error) {
        console.error('Error initializing 3Dmol viewer:', error)
        container.innerHTML = `
            <div class="text-center p-8">
                <div class="text-red-600 mb-2">
                    <i class="fas fa-exclamation-triangle text-4xl"></i>
                </div>
                <p class="text-sm text-gray-500">Error loading 3D viewer</p>
            </div>
        `
    }
}

// 返回上一页
const goBack = () => {
    router.back()
}

// 清理资源
onUnmounted(() => {
    // 清理3Dmol查看器（如果有）
    if (viewer3d) {
        try {
            viewer3d.clear()
            viewer3d = null
        } catch (error) {
            console.error('Error cleaning up 3Dmol viewer:', error)
        }
    }

    // 清理 sessionStorage
    sessionStorage.removeItem('currentMolecule')
})
</script>

<style scoped>
/* 自定义样式 */
#molecule-2d,
#molecule-3d,
#radar-chart {
    position: relative;
}

/* 圆圈动画 */
.bg-red-500,
.bg-green-500 {
    animation: pulse 2s infinite;
}

@keyframes pulse {

    0%,
    100% {
        opacity: 1;
    }

    50% {
        opacity: 0.7;
    }
}
</style>
