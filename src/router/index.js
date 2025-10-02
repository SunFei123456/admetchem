import { createRouter, createWebHistory,createWebHashHistory } from 'vue-router'
import Home from '@/views/Home.vue'
import About from '@/views/About.vue'
import DruglikenessEvaluation from '@/views/DruglikenessEvaluation.vue'
import DruglikenessResult from '@/views/DruglikenessResult.vue'
import AdmetPrediction from '@/views/AdmetPrediction.vue'
import SystematicEvaluation from '@/views/SystematicEvaluation.vue'
import ApplicationDomain from '@/views/ApplicationDomain.vue'
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
            path: '/systematic-evaluation',
            name: 'SystematicEvaluation',
            component: SystematicEvaluation
        },
        {
            path: '/application-domain',
            name: 'ApplicationDomain',
            component: ApplicationDomain
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