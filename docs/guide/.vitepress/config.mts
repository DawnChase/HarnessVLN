import { defineConfig } from 'vitepress'

export default defineConfig({
  lang: 'zh-CN',
  title: 'HarnessVLN Guide',
  description: 'Agent 主导的模块化视觉语言导航 Harness 架构与开发指南',
  outDir: '../../page',
  cleanUrls: false,
  lastUpdated: true,
  head: [
    ['meta', { name: 'theme-color', content: '#176b58' }],
    ['link', { rel: 'icon', type: 'image/png', href: '/harnessvln-mark.png' }],
  ],
  markdown: {
    lineNumbers: true,
  },
  themeConfig: {
    logo: '/harnessvln-mark.png',
    siteTitle: 'HarnessVLN',
    nav: [
      { text: '开始', link: '/getting-started/mental-model' },
      { text: '架构', link: '/architecture/overview' },
      { text: '组件', link: '/components/agents' },
      { text: '开发', link: '/extending/plugin-contract' },
      { text: '参考', link: '/reference/tool-catalog' },
    ],
    sidebar: [
      {
        text: '导读',
        items: [
          { text: '文档首页', link: '/' },
          { text: '先建立正确心智模型', link: '/getting-started/mental-model' },
          { text: '十分钟运行', link: '/getting-started/quick-start' },
          { text: '术语与边界', link: '/getting-started/concepts' },
          { text: '仓库地图', link: '/getting-started/repository-map' },
        ],
      },
      {
        text: '架构',
        items: [
          { text: '总体分层', link: '/architecture/overview' },
          { text: 'Task 与 Goal 模型', link: '/architecture/task-model' },
          { text: '一次任务如何运行', link: '/architecture/execution-flow' },
          { text: '并行与生命周期', link: '/architecture/concurrency-lifecycle' },
          { text: '数据与控制流', link: '/architecture/data-control-flow' },
        ],
      },
      {
        text: '核心组件',
        items: [
          { text: 'Agent Core', link: '/components/agents' },
          { text: 'VLN 插件', link: '/components/vln' },
          { text: 'Environment 中间件', link: '/components/environments' },
          { text: 'Bench 与评分', link: '/components/benchmarks' },
          { text: 'ToolBus 与函数调用', link: '/components/tool-bus' },
          { text: '空间记忆', link: '/components/memory' },
        ],
      },
      {
        text: '使用',
        items: [
          { text: '配置叠加', link: '/usage/configuration' },
          { text: '运行 Bench', link: '/usage/running-benchmarks' },
          { text: '结果与 Manifest', link: '/usage/results-manifest' },
          { text: '数据、模型与环境', link: '/usage/data-model-environment' },
        ],
      },
      {
        text: '扩展开发',
        items: [
          { text: '插件契约总览', link: '/extending/plugin-contract' },
          { text: '添加 Agent', link: '/extending/add-agent' },
          { text: '添加 VLN', link: '/extending/add-vln' },
          { text: '添加 Environment 与 Bench', link: '/extending/add-environment-bench' },
          { text: '添加空间记忆', link: '/extending/add-memory' },
        ],
      },
      {
        text: '参考与运维',
        items: [
          { text: '工具目录', link: '/reference/tool-catalog' },
          { text: '兼容与验证矩阵', link: '/reference/compatibility' },
          { text: '测试与发布门禁', link: '/reference/testing-validation' },
          { text: '故障定位', link: '/reference/troubleshooting' },
          { text: '设计取舍', link: '/reference/design-decisions' },
        ],
      },
    ],
    search: {
      provider: 'local',
    },
    outline: {
      level: [2, 3],
      label: '本页目录',
    },
    docFooter: {
      prev: '上一页',
      next: '下一页',
    },
    lastUpdated: {
      text: '最后更新',
      formatOptions: {
        dateStyle: 'medium',
        timeStyle: 'short',
      },
    },
    returnToTopLabel: '返回顶部',
    sidebarMenuLabel: '文档导航',
    darkModeSwitchLabel: '外观',
    lightModeSwitchTitle: '切换到浅色模式',
    darkModeSwitchTitle: '切换到深色模式',
    footer: {
      message: 'Agent-led navigation harness · 文档源位于 docs/guide',
      copyright: 'HarnessVLN',
    },
  },
})
