import { createRouter, createWebHistory,createWebHashHistory } from 'vue-router'
import Home from '@/views/Home.vue'
import About from '@/views/About.vue'
import DruglikenessEvaluation from '@/views/DruglikenessEvaluation.vue'
import DruglikenessResult from '@/views/DruglikenessResult.vue'
import AdmetPrediction from '@/views/AdmetPrediction.vue'
import AdmetPredictionResult from '@/views/AdmetPredictionResult.vue'
import MoleculeDetail from '@/views/MoleculeDetail.vue'
import Optimization from '@/views/Optimization.vue'
import Search from '@/views/Search.vue'

const router = createRouter({
    history: createWebHashHistory(),
    routes: [
        {
            path: '/',
            name: 'Home',
            component: Home
        },
        {
            path: '/about',
            name: 'About',
            component: About
        },
        {
            path: '/druglikeness-evaluation',
            name: 'DruglikenessEvaluation',
            component: DruglikenessEvaluation
        },
        {
            path: '/druglikeness-result',
            name: 'DruglikenessResult',
            component: DruglikenessResult
        },
        {
            path: '/admet-prediction',
            name: 'AdmetPrediction',
            component: AdmetPrediction
        },
        {
            path: '/admet-prediction-result',
            name: 'admet-prediction-result',
            component: AdmetPredictionResult
        },
        {
            path: '/molecule-detail',
            name: 'molecule-detail',
            component: MoleculeDetail
        },
        {
            path: '/optimization',
            name: 'Optimization',
            component: Optimization
        },
        {
            path: '/search',
            name: 'Search',
            component: Search
        }
    ]
})

export default router