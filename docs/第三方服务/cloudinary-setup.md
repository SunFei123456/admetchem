# Cloudinary 图片上传服务配置指南

## 📋 前置条件

1. 注册 Cloudinary 账号：https://cloudinary.com/users/register_free
2. 登录 Cloudinary Dashboard：https://console.cloudinary.com/

## 🔧 配置步骤

### 1. 获取 Cloud Name

1. 登录 Cloudinary Dashboard
2. 在 Dashboard 首页找到 **Account Details** 部分
3. 复制 **Cloud Name**（例如：`dazdjqzwd`）
4. 这个值已经配置在 `src/utils/cloudinary.js` 中

### 2. 创建 Upload Preset（重要！）

Upload Preset 是允许未授权上传的配置，不需要暴露 API Secret。

#### 步骤：

1. 在 Cloudinary Dashboard 中，导航到：
   ```
   Settings (⚙️) → Upload → Upload presets
   ```

2. 点击 **Add upload preset** 按钮

3. 配置 Upload Preset：

   **基本设置：**
   - **Preset name**: `admet_avatars`（必须与代码中的 `uploadPreset` 一致）
   - **Signing mode**: 选择 **Unsigned** ✅（允许客户端直接上传）
   - **Folder**: `admet/avatars`（可选，组织文件）
   - **Use filename or externally defined Public ID**: 选择 **Yes**

   **上传限制：**
   - **Resource type**: `Image`
   - **Allowed formats**: `jpg, jpeg, png, gif, webp`
   - **Max file size**: `3MB` (3145728 bytes)
   - **Max image width**: `800` pixels
   - **Max image height**: `800` pixels

   **图片转换（可选但推荐）：**
   - **Incoming Transformation**:
     - Width: `800`
     - Height: `800`
     - Crop: `fill`
     - Quality: `auto`
     - Format: `auto`
   
   这样可以自动优化上传的图片，减少存储空间。

4. 点击 **Save** 保存配置

### 3. 更新代码配置

如果你使用了不同的 Cloud Name 或 Upload Preset 名称，需要更新 `src/utils/cloudinary.js`：

```javascript
const CLOUDINARY_CONFIG = {
  cloudName: 'your-cloud-name', // 👈 更新为你的 Cloud Name
  uploadPreset: 'your-preset-name', // 👈 更新为你的 Upload Preset 名称
  // ... 其他配置
}
```

## 🚀 使用方法

### 在 Profile 页面上传头像

1. 访问个人中心页面（Profile）
2. 在 **Avatar** 部分，点击绿色的 **Upload** 按钮
3. 选择上传方式：
   - **My Files**: 从本地上传
   - **Web Address**: 输入图片URL
   - **Camera**: 使用摄像头拍照
4. 如果启用了裁剪，会显示裁剪界面（正方形 1:1）
5. 点击 **Crop & Upload** 完成上传
6. 上传成功后，图片 URL 会自动填充到输入框
7. 点击 **Save Changes** 保存到数据库

### 也可以手动输入 URL

如果你已经有图片 URL，可以直接粘贴到输入框中，无需上传。

## 🔒 安全性说明

### Unsigned Upload 的安全性

使用 **Unsigned** 模式是安全的，因为：

1. ✅ **不暴露 API Secret**：客户端不需要知道 API Secret
2. ✅ **Upload Preset 限制**：在后台配置了严格的上传限制（文件大小、格式、尺寸等）
3. ✅ **Cloudinary 自动检测**：自动检测和阻止恶意文件
4. ✅ **速率限制**：Cloudinary 有内置的速率限制防止滥用

### 推荐的安全措施

如果担心被滥用，可以：

1. **启用 Moderation（内容审核）**：
   - 在 Upload Preset 中启用 `Manual` 或 `WebPurify` moderation
   - 需要手动批准或自动过滤不当内容

2. **限制来源域名**：
   - Settings → Security → Allowed domains
   - 只允许特定域名调用上传 API

3. **监控上传量**：
   - Dashboard → Reports
   - 定期检查上传量，发现异常及时处理

4. **使用 Cloudinary Webhooks**：
   - 上传成功后触发 webhook
   - 在后端验证和记录上传行为

## 📊 Cloudinary 免费套餐限制

| 项目 | 限制 |
|------|------|
| 存储空间 | 25 GB |
| 带宽 | 25 GB/月 |
| 转换次数 | 25,000 次/月 |
| 上传数量 | 无限制 |

对于个人项目和小型应用完全够用！

## 🆘 常见问题

### Q: 上传时提示 "Upload preset not found"？
**A**: 检查 `uploadPreset` 名称是否与 Cloudinary 后台配置的一致。

### Q: 上传时提示 "Unsigned upload is not allowed"？
**A**: 确保 Upload Preset 的 **Signing mode** 设置为 **Unsigned**。

### Q: 图片上传成功但很慢？
**A**: 
- 检查图片大小，建议上传前压缩
- 使用 Cloudinary 的自动转换功能
- 配置 CDN 加速

### Q: 想要更改上传后的图片质量/格式？
**A**: 在 Upload Preset 的 **Incoming Transformation** 中配置：
- Quality: `auto` 或具体值（1-100）
- Format: `auto`（自动选择最优格式）或指定格式

### Q: 如何删除已上传的图片？
**A**: 
- 在 Cloudinary Dashboard → Media Library 中手动删除
- 或使用 Admin API（需要后端实现）

## 🔗 相关链接

- [Cloudinary Upload Widget 文档](https://cloudinary.com/documentation/upload_widget)
- [Upload Presets 文档](https://cloudinary.com/documentation/upload_presets)
- [Unsigned Upload 文档](https://cloudinary.com/documentation/upload_images#unsigned_upload)
- [图片优化最佳实践](https://cloudinary.com/documentation/image_optimization)

## ✅ 完成检查清单

- [ ] 注册 Cloudinary 账号
- [ ] 复制 Cloud Name 并更新到代码
- [ ] 创建 Upload Preset（名称：`admet_avatars`，模式：Unsigned）
- [ ] 配置上传限制（文件大小、格式、尺寸）
- [ ] （可选）配置图片自动转换
- [ ] 测试上传功能
- [ ] 检查上传的图片 URL 是否正确保存到数据库

---

**配置完成后，就可以在个人中心页面愉快地上传头像了！** 🎉

