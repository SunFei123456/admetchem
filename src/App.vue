<template>
  <div id="app">
    <Navigation />
    <Breadcrumb />
    <router-view />
  </div>
</template>

<script setup>
import Navigation from '@/components/Navigation.vue'
import Breadcrumb from '@/components/Breadcrumb.vue'
</script>
