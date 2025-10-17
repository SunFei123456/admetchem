/**
 * Cloudinary 图片上传服务配置
 * 文档: https://cloudinary.com/documentation/upload_widget
 */

// Cloudinary 配置
const CLOUDINARY_CONFIG = {
  cloudName: 'dazdjqzwd', //  Cloud Name（从你提供的默认头像 URL 中提取）
  uploadPreset: 'admet_avatars', // 需要在 Cloudinary 后台创建 Upload Preset
  folder: 'admet/avatars', // 上传到的文件夹
  maxFileSize: 5000000, // 5MB
  maxImageWidth: 800,
  maxImageHeight: 800,
  cropping: false, // 启用裁剪
  croppingAspectRatio: 1, // 1:1 正方形头像
  croppingShowDimensions: true,
  multiple: false, // 只允许单个文件
  resourceType: 'image',
  clientAllowedFormats: ['jpg', 'jpeg', 'png', 'gif', 'webp'],
  sources: ['local', 'url', 'camera'], // 允许的上传源
  showSkipCropButton: false,
  croppingCoordinatesMode: 'custom',
  language: 'en', // 界面语言
  text: {
    en: {
      or: 'Or',
      menu: {
        files: 'My Files',
        url: 'Web Address',
        camera: 'Camera'
      },
      crop: {
        title: 'Crop your avatar',
        crop_btn: 'Crop & Upload',
        skip_btn: 'Skip',
        reset_btn: 'Reset'
      },
      queue: {
        title: 'Upload Queue',
        title_uploading_with_counter: 'Uploading {{num}} Assets',
        title_processing_with_counter: 'Processing {{num}} Assets',
        done: 'Done',
        abort_all: 'Abort all',
        mini_title: 'Uploaded',
        mini_title_uploading: 'Uploading',
        mini_title_processing: 'Processing',
        statuses: {
          uploading: 'Uploading...',
          processing: 'Processing...',
          error: 'Error',
          uploaded: 'Uploaded',
          aborted: 'Aborted'
        }
      },
      local: {
        browse: 'Browse',
        dd_title_single: 'Drag and Drop an image here',
        dd_title_multi: 'Drag and Drop images here',
        drop_title_single: 'Drop an image to upload',
        drop_title_multiple: 'Drop images to upload'
      }
    }
  }
}

/**
 * 加载 Cloudinary Upload Widget 脚本
 * @returns {Promise<void>}
 */
export function loadCloudinaryScript() {
  return new Promise((resolve, reject) => {
    // 检查是否已经加载
    if (window.cloudinary) {
      resolve()
      return
    }

    // 创建 script 标签
    const script = document.createElement('script')
    script.src = 'https://upload-widget.cloudinary.com/global/all.js'
    script.async = true
    script.onload = () => resolve()
    script.onerror = () => reject(new Error('Failed to load Cloudinary script'))
    
    document.head.appendChild(script)
  })
}

/**
 * 打开 Cloudinary 上传 Widget
 * @param {Object} options - 额外的配置选项
 * @param {Function} onSuccess - 上传成功回调 (url, publicId, resource) => void
 * @param {Function} onError - 上传失败回调 (error) => void
 * @returns {Object} widget instance
 */
export function openUploadWidget(options = {}, onSuccess, onError) {
  if (!window.cloudinary) {
    onError?.(new Error('Cloudinary script not loaded'))
    return null
  }

  const config = {
    ...CLOUDINARY_CONFIG,
    ...options
  }

  const widget = window.cloudinary.createUploadWidget(
    config,
    (error, result) => {
      if (error) {
        console.error('Cloudinary upload error:', error)
        onError?.(error)
        return
      }

      // 上传成功
      if (result.event === 'success') {
        const { secure_url, public_id, resource_type } = result.info
        console.log('Upload successful:', {
          url: secure_url,
          publicId: public_id,
          resourceType: resource_type
        })
        onSuccess?.(secure_url, public_id, result.info)
      }

      // 上传完成（所有文件）
      if (result.event === 'queues-end') {
        console.log('All uploads completed')
      }
    }
  )

  widget.open()
  return widget
}

/**
 * 上传单个图片并返回 URL（Promise 版本）
 * @param {Object} options - 额外的配置选项
 * @returns {Promise<{url: string, publicId: string}>}
 */
export function uploadImage(options = {}) {
  return new Promise((resolve, reject) => {
    openUploadWidget(
      options,
      (url, publicId) => {
        resolve({ url, publicId })
      },
      (error) => {
        reject(error)
      }
    )
  })
}

export default {
  loadCloudinaryScript,
  openUploadWidget,
  uploadImage,
  CLOUDINARY_CONFIG
}

