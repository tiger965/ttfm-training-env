# 🚀 GitHub推送指南

## 快速推送步骤

### 1. 创建GitHub Personal Access Token

1. 访问: https://github.com/settings/tokens
2. 点击 "Generate new token (classic)"
3. 设置:
   - Note: `ttfm-training-env`
   - Expiration: 90 days
   - 权限勾选:
     - ✅ repo (全部)
     - ✅ workflow
4. 点击 "Generate token"
5. **复制token（只显示一次！）**

### 2. 创建仓库

访问: https://github.com/new

填写:
- Repository name: `ttfm-training-env`
- Description: `RTX 50系列深度学习环境 - 开箱即用 | RTX 50 Series Deep Learning Environment - One-Click Setup`
- Public ✅
- 不要勾选 "Initialize this repository with a README"

点击 "Create repository"

### 3. 推送代码

```bash
cd /mnt/d/GitRepos/ttfm-training-env

# 设置远程仓库
git remote add origin https://github.com/tiger965/ttfm-training-env.git

# 推送（会要求输入用户名和token）
git push -u origin main

# 输入:
# Username: tiger965
# Password: [粘贴你的token]
```

### 4. 或者使用token直接推送

```bash
# 一次性命令（替换YOUR_TOKEN）
git push https://tiger965:YOUR_TOKEN@github.com/tiger965/ttfm-training-env.git main
```

## 📝 推送后记得

1. 添加项目描述
2. 添加Topics标签：
   - `rtx-5080`
   - `rtx-5090`
   - `deep-learning`
   - `pytorch`
   - `sm-120`
   - `cuda`

3. 在README添加徽章:
```markdown
[![GitHub stars](https://img.shields.io/github/stars/tiger965/ttfm-training-env)](https://github.com/tiger965/ttfm-training-env/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/tiger965/ttfm-training-env)](https://github.com/tiger965/ttfm-training-env/issues)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
```

## 🎯 完成！

推送成功后，全世界的RTX 50用户都能使用了！