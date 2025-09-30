# Vue 3 + Vue Router + Tailwind CSS + Vite 项目

一个现代化的 Vue 3 全栈项目模板，集成了 Vue Router、Tailwind CSS 和 Vite。

## 🚀 技术栈

- **Vue 3** - 渐进式 JavaScript 框架
- **Vue Router 4** - 官方路由管理器
- **Tailwind CSS** - 实用优先的 CSS 框架
- **Vite** - 下一代前端构建工具
- **pnpm** - 快速、节省磁盘空间的包管理器

## 📁 项目结构

```
src/
├── api/          # API 相关文件
├── assets/       # 静态资源
├── components/   # Vue 组件
├── constants/    # 常量配置
├── router/       # 路由配置
├── utils/        # 工具函数
├── views/        # 页面组件
├── App.vue       # 根组件
├── main.js       # 入口文件
└── style.css     # 全局样式
```

## 🛠️ 开发指南

### 环境要求

- Node.js 20.19+ 或 22.12+
- pnpm (推荐) 或 npm

### 安装依赖

```bash
pnpm install
```

### 启动开发服务器

```bash
pnpm run dev
```

### 构建生产版本

```bash
pnpm run build
```

### 预览生产版本

```bash
pnpm run preview
```

## 🎯 功能特性

- ✅ Vue 3 Composition API
- ✅ `<script setup>` 语法糖
- ✅ Vue Router 4 路由管理
- ✅ Tailwind CSS 原子化样式
- ✅ Vite 快速热更新
- ✅ 路径别名配置
- ✅ 响应式设计
- ✅ 现代化 UI 组件

## 📋 别名配置

项目配置了以下路径别名：

- `@/` → `src/`
- `@/components` → `src/components/`
- `@/views` → `src/views/`
- `@/utils` → `src/utils/`
- `@/api` → `src/api/`
- `@/assets` → `src/assets/`
- `@/router` → `src/router/`
- `@/constants` → `src/constants/`

## 🔧 推荐 IDE 配置

- [VS Code](https://code.visualstudio.com/)
- [Vue - Official](https://marketplace.visualstudio.com/items?itemName=Vue.volar) 扩展
- [Tailwind CSS IntelliSense](https://marketplace.visualstudio.com/items?itemName=bradlc.vscode-tailwindcss) 扩展
