import { defineConfig } from "vitepress";

export default defineConfig({
  title: "Liuls Notes",
  description: "个人学习笔记",
  lang: "zh-CN",
  base: process.env.BASE_PATH || "/",
  cleanUrls: true,
  lastUpdated: true,
  markdown: {
    math: true
  },
  themeConfig: {
    logo: undefined,
    nav: [
      { text: "首页", link: "/" },
      { text: "强化学习", link: "/RL-Algorithm/0-基础" },
      { text: "工具", link: "/Other/git命令" }
    ],
    sidebar: {
      "/RL-Algorithm/": [
        {
          text: "强化学习",
          items: [
            { text: "0. 基础", link: "/RL-Algorithm/0-基础" },
            { text: "1. DQN", link: "/RL-Algorithm/1-DQN" },
            { text: "2. REINFORCE", link: "/RL-Algorithm/2-REINFORCE" },
            { text: "3. Actor-Critic", link: "/RL-Algorithm/3-Actor-Critic" },
            { text: "4. TRPO", link: "/RL-Algorithm/4-TRPO" },
            { text: "5. PPO", link: "/RL-Algorithm/5-PPO" },
            { text: "6. DPO", link: "/RL-Algorithm/6-DPO" },
            { text: "7. GRPO 系列", link: "/RL-Algorithm/7-GRPO系列" }
          ]
        }
      ],
      "/Other/": [
        {
          text: "工具",
          items: [
            { text: "Git 命令", link: "/Other/git命令" },
            { text: "VS Code", link: "/Other/vscode" }
          ]
        }
      ]
    },
    socialLinks: [],
    outline: {
      level: [2, 3],
      label: "本页目录"
    },
    docFooter: {
      prev: "上一篇",
      next: "下一篇"
    },
    lastUpdated: {
      text: "最后更新",
      formatOptions: {
        dateStyle: "medium",
        timeStyle: "short"
      }
    },
    search: {
      provider: "local",
      options: {
        translations: {
          button: {
            buttonText: "搜索",
            buttonAriaLabel: "搜索"
          },
          modal: {
            displayDetails: "显示详情",
            resetButtonTitle: "清除搜索",
            backButtonTitle: "关闭搜索",
            noResultsText: "没有找到结果",
            footer: {
              selectText: "选择",
              selectKeyAriaLabel: "enter",
              navigateText: "切换",
              navigateUpKeyAriaLabel: "up arrow",
              navigateDownKeyAriaLabel: "down arrow",
              closeText: "关闭",
              closeKeyAriaLabel: "escape"
            }
          }
        }
      }
    }
  }
});
