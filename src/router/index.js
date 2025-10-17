import { createRouter, createWebHistory, createWebHashHistory } from 'vue-router'
import Home from '@/views/Home.vue'
import About from '@/views/About.vue'
import DruglikenessEvaluation from '@/views/DruglikenessEvaluation.vue'
import DruglikenessResult from '@/views/DruglikenessResult.vue'
import AdmetPrediction from '@/views/AdmetPrediction.vue'
import AdmetPredictionResult from '@/views/AdmetPredictionResult.vue'
import MoleculeDetail from '@/views/MoleculeDetail.vue'
import Optimization from '@/views/Optimization.vue'
import Search from '@/views/Search.vue'
import Login from '@/views/Login.vue'
import Register from '@/views/Register.vue'
import Profile from '@/views/Profile.vue'
import { isLoggedIn } from '@/utils/auth.js'

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
            path: '/login',
            name: 'Login',
            component: Login,
            meta: { hideForAuth: true } // 已登录用户隐藏
        },
        {
            path: '/register',
            name: 'Register',
            component: Register,
            meta: { hideForAuth: true } // 已登录用户隐藏
        },
        {
            path: '/profile',
            name: 'Profile',
            component: Profile,
            meta: { requiresAuth: true } // 需要登录
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

// 路由守卫
router.beforeEach((to, from, next) => {
    const loggedIn = isLoggedIn()

    // 需要登录的页面
    if (to.meta.requiresAuth && !loggedIn) {
        next('/login')
        return
    }

    // 已登录用户访问登录/注册页面，重定向到首页
    if (to.meta.hideForAuth && loggedIn) {
        next('/')
        return
    }

    next()
})

export default router