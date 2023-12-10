package main

import (
	"cmp"
	"flag"
	"fmt"
	"image"
	"log"
	"math"
	"runtime"
	"time"
	"unsafe"

	// Used for decoding textures
	"image/draw"
	_ "image/jpeg"

	"github.com/ironsmile/vulkan-tutorial-go/code/27_shader_depth/shaders"
	"github.com/ironsmile/vulkan-tutorial-go/code/optional"
	"github.com/ironsmile/vulkan-tutorial-go/code/unsafer"
	"github.com/ironsmile/vulkan-tutorial-go/textures"

	"github.com/go-gl/glfw/v3.3/glfw"
	vk "github.com/vulkan-go/vulkan"
	"github.com/xlab/linmath"
)

func init() {
	// This is needed to arrange that main() runs on main thread.
	// See documentation for functions that are only allowed to be called
	// from the main thread.
	runtime.LockOSThread()

	flag.BoolVar(&args.debug, "debug", false, "Enable Vulkan validation layers")
}

var args struct {
	debug bool
}

const (
	title             = "Vulkan Tutorial: Depth buffering"
	maxFramesInFlight = 2
)

func main() {
	flag.Parse()

	app := &VulkanTutorialApp{
		width:       1024,
		height:      768,
		texturePath: "texture.jpg",

		enableValidationLayers: args.debug,
		startTime:              time.Now(),
		validationLayers: []string{
			"VK_LAYER_KHRONOS_validation\x00",
		},
		deviceExtensions: []string{
			vk.KhrSwapchainExtensionName + "\x00",
		},
		physicalDevice: vk.PhysicalDevice(vk.NullHandle),
		device:         vk.Device(vk.NullHandle),
		surface:        vk.NullSurface,
		swapChain:      vk.NullSwapchain,
		vertices: []Vertex{
			{
				pos:      linmath.Vec3{-0.5, -0.5, 0},
				color:    linmath.Vec3{1, 0, 0},
				texColor: linmath.Vec2{1, 0},
			},
			{
				pos:      linmath.Vec3{0.5, -0.5, 0},
				color:    linmath.Vec3{0, 1, 0},
				texColor: linmath.Vec2{0, 0},
			},
			{
				pos:      linmath.Vec3{0.5, 0.5, 0},
				color:    linmath.Vec3{0, 0, 1},
				texColor: linmath.Vec2{0, 1},
			},
			{
				pos:      linmath.Vec3{-0.5, 0.5, 0},
				color:    linmath.Vec3{1, 1, 1},
				texColor: linmath.Vec2{1, 1},
			},

			{
				pos:      linmath.Vec3{-0.5, -0.5, -0.5},
				color:    linmath.Vec3{1, 0, 0},
				texColor: linmath.Vec2{1, 0},
			},
			{
				pos:      linmath.Vec3{0.5, -0.5, -0.5},
				color:    linmath.Vec3{0, 1, 0},
				texColor: linmath.Vec2{0, 0},
			},
			{
				pos:      linmath.Vec3{0.5, 0.5, -0.5},
				color:    linmath.Vec3{0, 0, 1},
				texColor: linmath.Vec2{0, 1},
			},
			{
				pos:      linmath.Vec3{-0.5, 0.5, -0.5},
				color:    linmath.Vec3{1, 1, 1},
				texColor: linmath.Vec2{1, 1},
			},
		},
		indices: []uint16{
			0, 1, 2, 2, 3, 0,
			4, 5, 6, 6, 7, 4,
		},
		vertexBuffer:       vk.NullBuffer,
		vertexBufferMemory: vk.NullDeviceMemory,
		indexBuffer:        vk.NullBuffer,
		indexBufferMemory:  vk.NullDeviceMemory,
		descriptorPool:     vk.NullDescriptorPool,

		textureImage:       vk.NullImage,
		textureImageMemory: vk.NullDeviceMemory,
		textureSampler:     vk.NullSampler,

		descriptorSetLayout: vk.NullDescriptorSetLayout,
	}
	if err := app.Run(); err != nil {
		log.Fatalf("ERROR: %s", err)
	}
}

// VulkanTutorialApp is the first program from vulkan-tutorial.com
type VulkanTutorialApp struct {
	width  int
	height int

	texturePath string

	// validationLayers is the list of required device extensions needed by this
	// program when the -debug flag is set.
	validationLayers       []string
	enableValidationLayers bool

	// deviceExtensions is the list of required device extensions needed by this
	// program.
	deviceExtensions []string

	window   *glfw.Window
	instance vk.Instance

	// physicalDevice is the physical device selected for this program.
	physicalDevice vk.PhysicalDevice

	// device is the logical device created for interfacing with the physical device.
	device vk.Device

	startTime time.Time

	graphicsQueue vk.Queue
	presentQueue  vk.Queue

	surface vk.Surface

	swapChain            vk.Swapchain
	swapChainImages      []vk.Image
	swapChainImageViews  []vk.ImageView
	swapChainImageFormat vk.Format
	swapChainExtend      vk.Extent2D

	swapChainFramebuffers []vk.Framebuffer

	renderPass          vk.RenderPass
	descriptorSetLayout vk.DescriptorSetLayout
	pipelineLayout      vk.PipelineLayout

	graphicsPipline vk.Pipeline

	commandPool    vk.CommandPool
	commandBuffers []vk.CommandBuffer

	imageAvailabmeSems []vk.Semaphore
	renderFinishedSems []vk.Semaphore
	inFlightFences     []vk.Fence

	frameBufferResized bool

	curentFrame uint32

	vertices           []Vertex
	vertexBuffer       vk.Buffer
	vertexBufferMemory vk.DeviceMemory

	indices           []uint16
	indexBuffer       vk.Buffer
	indexBufferMemory vk.DeviceMemory

	uniformBuffers       []vk.Buffer
	uniformBuffersMemory []vk.DeviceMemory
	uniformBuffersMapped []unsafe.Pointer

	descriptorPool vk.DescriptorPool
	descriptorSets []vk.DescriptorSet

	textureImage       vk.Image
	textureImageMemory vk.DeviceMemory
	textureImageView   vk.ImageView
	textureSampler     vk.Sampler

	depthImage       vk.Image
	depthImageMemory vk.DeviceMemory
	depthImageView   vk.ImageView
}

// Run runs the vulkan program.
func (a *VulkanTutorialApp) Run() error {
	if err := a.initWindow(); err != nil {
		return fmt.Errorf("initWindow: %w", err)
	}
	defer a.cleanWindow()

	if err := a.initVulkan(); err != nil {
		return fmt.Errorf("initVulkan: %w", err)
	}
	defer a.cleanupVulkan()

	if err := a.mainLoop(); err != nil {
		return fmt.Errorf("mainLoop: %w", err)
	}

	return nil
}

func (a *VulkanTutorialApp) initWindow() error {
	if err := glfw.Init(); err != nil {
		return fmt.Errorf("glfw.Init: %w", err)
	}

	glfw.WindowHint(glfw.ClientAPI, glfw.NoAPI)

	window, err := glfw.CreateWindow(a.width, a.height, title, nil, nil)
	if err != nil {
		return fmt.Errorf("creating window: %w", err)
	}

	window.SetFramebufferSizeCallback(a.frameBufferResizeCallback)

	a.window = window
	return nil
}

func (a *VulkanTutorialApp) frameBufferResizeCallback(
	w *glfw.Window,
	width int,
	height int,
) {
	a.frameBufferResized = true
}

func (a *VulkanTutorialApp) cleanWindow() {
	a.window.Destroy()
	glfw.Terminate()
}

func (a *VulkanTutorialApp) initVulkan() error {
	vk.SetGetInstanceProcAddr(glfw.GetVulkanGetInstanceProcAddress())

	if err := vk.Init(); err != nil {
		return fmt.Errorf("failed to init Vulkan Go: %w", err)
	}

	if err := a.createInsance(); err != nil {
		return fmt.Errorf("createInstance: %w", err)
	}

	if err := a.createSurface(); err != nil {
		return fmt.Errorf("createSurface: %w", err)
	}

	if err := a.pickPhysicalDevice(); err != nil {
		return fmt.Errorf("pickPhysicalDevice: %w", err)
	}

	if err := a.createLogicalDevice(); err != nil {
		return fmt.Errorf("createLogicalDevice: %w", err)
	}

	if err := a.createSwapChain(); err != nil {
		return fmt.Errorf("createSwapChain: %w", err)
	}

	if err := a.createImageViews(); err != nil {
		return fmt.Errorf("createImageViews: %w", err)
	}

	if err := a.createRenderPass(); err != nil {
		return fmt.Errorf("createRenderPass: %w", err)
	}

	if err := a.createDescriptorSetLayout(); err != nil {
		return fmt.Errorf("createDescriptorSetLayout: %w", err)
	}

	if err := a.createGraphicsPipeline(); err != nil {
		return fmt.Errorf("createGraphicsPipeline: %w", err)
	}

	if err := a.createCommandPool(); err != nil {
		return fmt.Errorf("createCommandPool: %w", err)
	}

	if err := a.createDepthResources(); err != nil {
		return fmt.Errorf("createDepthResources: %w", err)
	}

	if err := a.createFramebuffers(); err != nil {
		return fmt.Errorf("createFramebuffers: %w", err)
	}

	if err := a.createTextureImage(); err != nil {
		return fmt.Errorf("createTextureImage: %w", err)
	}

	if err := a.createTextureImageView(); err != nil {
		return fmt.Errorf("createTextureImageView: %w", err)
	}

	if err := a.createTextureSampler(); err != nil {
		return fmt.Errorf("createTextureSampler: %w", err)
	}

	if err := a.createVertexBuffer(); err != nil {
		return fmt.Errorf("createVertexBuffer: %w", err)
	}

	if err := a.createIndexBuffer(); err != nil {
		return fmt.Errorf("createIndexBuffer: %w", err)
	}

	if err := a.createUniformBuffers(); err != nil {
		return fmt.Errorf("createUniformBuffers: %w", err)
	}

	if err := a.createDescriptorPool(); err != nil {
		return fmt.Errorf("createDescriptorPool: %w", err)
	}

	if err := a.createDescriptorSets(); err != nil {
		return fmt.Errorf("createDescriptorSets: %w", err)
	}

	if err := a.createCommandBuffer(); err != nil {
		return fmt.Errorf("createCommandBuffer: %w", err)
	}

	if err := a.createSyncObjects(); err != nil {
		return fmt.Errorf("createSyncObjects: %w", err)
	}

	return nil
}

func (a *VulkanTutorialApp) cleanupVulkan() {
	for i := 0; i < maxFramesInFlight; i++ {
		vk.DestroySemaphore(a.device, a.imageAvailabmeSems[i], nil)
		vk.DestroySemaphore(a.device, a.renderFinishedSems[i], nil)
		vk.DestroyFence(a.device, a.inFlightFences[i], nil)
	}

	vk.DestroyCommandPool(a.device, a.commandPool, nil)

	vk.DestroyPipeline(a.device, a.graphicsPipline, nil)
	vk.DestroyPipelineLayout(a.device, a.pipelineLayout, nil)

	a.cleanupSwapChain()

	if a.textureSampler != vk.NullSampler {
		vk.DestroySampler(a.device, a.textureSampler, nil)
	}

	if a.textureImageView != vk.NullImageView {
		vk.DestroyImageView(a.device, a.textureImageView, nil)
	}

	if a.textureImage != vk.NullImage {
		vk.DestroyImage(a.device, a.textureImage, nil)
	}
	if a.textureImageMemory != vk.NullDeviceMemory {
		vk.FreeMemory(a.device, a.textureImageMemory, nil)
	}

	for _, buffer := range a.uniformBuffers {
		vk.DestroyBuffer(a.device, buffer, nil)
	}
	for _, bufferMem := range a.uniformBuffersMemory {
		vk.FreeMemory(a.device, bufferMem, nil)
	}

	if a.descriptorPool != vk.NullDescriptorPool {
		vk.DestroyDescriptorPool(a.device, a.descriptorPool, nil)
	}

	if a.descriptorSetLayout != vk.NullDescriptorSetLayout {
		vk.DestroyDescriptorSetLayout(a.device, a.descriptorSetLayout, nil)
	}

	if a.vertexBuffer != vk.NullBuffer {
		vk.DestroyBuffer(a.device, a.vertexBuffer, nil)
	}
	if a.vertexBufferMemory != vk.NullDeviceMemory {
		vk.FreeMemory(a.device, a.vertexBufferMemory, nil)
	}

	if a.indexBuffer != vk.NullBuffer {
		vk.DestroyBuffer(a.device, a.indexBuffer, nil)
	}
	if a.indexBufferMemory != vk.NullDeviceMemory {
		vk.FreeMemory(a.device, a.indexBufferMemory, nil)
	}

	vk.DestroyRenderPass(a.device, a.renderPass, nil)

	if a.device != vk.Device(vk.NullHandle) {
		vk.DestroyDevice(a.device, nil)
	}
	if a.surface != vk.NullSurface {
		vk.DestroySurface(a.instance, a.surface, nil)
	}
	vk.DestroyInstance(a.instance, nil)
}

func (a *VulkanTutorialApp) cleanupSwapChain() {
	if a.depthImageView != vk.NullImageView {
		vk.DestroyImageView(a.device, a.depthImageView, nil)
	}

	if a.depthImage != vk.NullImage {
		vk.DestroyImage(a.device, a.depthImage, nil)
	}

	if a.depthImageMemory != vk.NullDeviceMemory {
		vk.FreeMemory(a.device, a.depthImageMemory, nil)
	}

	for _, frameBuffer := range a.swapChainFramebuffers {
		vk.DestroyFramebuffer(a.device, frameBuffer, nil)
	}

	for _, imageView := range a.swapChainImageViews {
		vk.DestroyImageView(a.device, imageView, nil)
	}

	if a.swapChain != vk.NullSwapchain {
		vk.DestroySwapchain(a.device, a.swapChain, nil)
	}
	a.swapChainImages = nil
	a.swapChainImageViews = nil
}

func (a *VulkanTutorialApp) createSurface() error {
	surfacePtr, err := a.window.CreateWindowSurface(a.instance, nil)
	if err != nil {
		return fmt.Errorf("cannot create surface within GLFW window: %w", err)
	}

	a.surface = vk.SurfaceFromPointer(surfacePtr)
	return nil
}

func (a *VulkanTutorialApp) pickPhysicalDevice() error {
	var deviceCount uint32
	err := vk.Error(vk.EnumeratePhysicalDevices(a.instance, &deviceCount, nil))
	if err != nil {
		return fmt.Errorf("failed to get the number of physical devices: %w", err)
	}
	if deviceCount == 0 {
		return fmt.Errorf("failed to find GPUs with Vulkan support")
	}

	pDevices := make([]vk.PhysicalDevice, deviceCount)
	err = vk.Error(vk.EnumeratePhysicalDevices(a.instance, &deviceCount, pDevices))
	if err != nil {
		return fmt.Errorf("failed to enumerate the physical devices: %w", err)
	}

	var (
		selectedDevice vk.PhysicalDevice
		score          uint32
	)

	for _, device := range pDevices {
		deviceScore := a.getDeviceScore(device)

		if deviceScore > score {
			selectedDevice = device
			score = deviceScore
		}
	}

	if selectedDevice == vk.PhysicalDevice(vk.NullHandle) {
		return fmt.Errorf("failed to find suitable physical devices")
	}

	a.physicalDevice = selectedDevice
	return nil
}

func (a *VulkanTutorialApp) createLogicalDevice() error {
	indices := a.findQueueFamilies(a.physicalDevice)
	if !indices.IsComplete() {
		return fmt.Errorf("createLogicalDevice called for physical device which does " +
			"have all the queues required by the program")
	}

	queueFamilies := make(map[uint32]struct{})
	queueFamilies[indices.Graphics.Get()] = struct{}{}
	queueFamilies[indices.Present.Get()] = struct{}{}

	queueCreateInfos := []vk.DeviceQueueCreateInfo{}

	for familyIndex := range queueFamilies {
		queueCreateInfos = append(
			queueCreateInfos,
			vk.DeviceQueueCreateInfo{
				SType:            vk.StructureTypeDeviceQueueCreateInfo,
				QueueFamilyIndex: familyIndex,
				QueueCount:       1,
				PQueuePriorities: []float32{1.0},
			},
		)
	}

	deviceFeatures := []vk.PhysicalDeviceFeatures{{
		SamplerAnisotropy: vk.True,
	}}

	createInfo := vk.DeviceCreateInfo{
		SType:            vk.StructureTypeDeviceCreateInfo,
		PEnabledFeatures: deviceFeatures,

		PQueueCreateInfos:    queueCreateInfos,
		QueueCreateInfoCount: uint32(len(queueCreateInfos)),

		EnabledExtensionCount:   uint32(len(a.deviceExtensions)),
		PpEnabledExtensionNames: a.deviceExtensions,
	}

	if a.enableValidationLayers {
		createInfo.PpEnabledLayerNames = a.validationLayers
		createInfo.EnabledLayerCount = uint32(len(a.validationLayers))
	}

	var device vk.Device
	err := vk.Error(vk.CreateDevice(a.physicalDevice, &createInfo, nil, &device))
	if err != nil {
		return fmt.Errorf("failed to create logical device: %w", err)
	}
	a.device = device

	var graphicsQueue vk.Queue
	vk.GetDeviceQueue(a.device, indices.Graphics.Get(), 0, &graphicsQueue)
	a.graphicsQueue = graphicsQueue

	var presentQueue vk.Queue
	vk.GetDeviceQueue(a.device, indices.Present.Get(), 0, &presentQueue)
	a.presentQueue = presentQueue

	return nil
}

func (a *VulkanTutorialApp) createSwapChain() error {
	swapChainSupport := a.querySwapChainSupport(a.physicalDevice)

	surfaceFormat := a.chooseSwapSurfaceFormat(swapChainSupport.formats)
	presentMode := a.chooseSwapPresentMode(swapChainSupport.presentModes)
	extend := a.chooseSwapExtend(swapChainSupport.capabilities)

	imageCount := swapChainSupport.capabilities.MinImageCount + 1
	if swapChainSupport.capabilities.MaxImageCount > 0 &&
		imageCount > swapChainSupport.capabilities.MaxImageCount {
		imageCount = swapChainSupport.capabilities.MaxImageCount
	}

	createInfo := vk.SwapchainCreateInfo{
		SType:            vk.StructureTypeSwapchainCreateInfo,
		Surface:          a.surface,
		MinImageCount:    imageCount,
		ImageColorSpace:  surfaceFormat.ColorSpace,
		ImageFormat:      surfaceFormat.Format,
		ImageExtent:      extend,
		ImageArrayLayers: 1,
		ImageUsage:       vk.ImageUsageFlags(vk.ImageUsageColorAttachmentBit),
		PreTransform:     swapChainSupport.capabilities.CurrentTransform,
		CompositeAlpha:   vk.CompositeAlphaOpaqueBit,
		PresentMode:      presentMode,
		Clipped:          vk.True,
	}

	indices := a.findQueueFamilies(a.physicalDevice)
	if indices.Graphics.Get() != indices.Present.Get() {
		createInfo.ImageSharingMode = vk.SharingModeConcurrent
		createInfo.QueueFamilyIndexCount = 2
		createInfo.PQueueFamilyIndices = []uint32{
			indices.Graphics.Get(),
			indices.Present.Get(),
		}
	} else {
		createInfo.ImageSharingMode = vk.SharingModeExclusive
	}

	var swapChain vk.Swapchain
	res := vk.CreateSwapchain(a.device, &createInfo, nil, &swapChain)
	if err := vk.Error(res); err != nil {
		return fmt.Errorf("failed to create swap chain: %w", err)
	}
	a.swapChain = swapChain

	var imagesCount uint32
	vk.GetSwapchainImages(a.device, a.swapChain, &imagesCount, nil)

	images := make([]vk.Image, imagesCount)
	vk.GetSwapchainImages(a.device, a.swapChain, &imagesCount, images)

	a.swapChainImages = images

	a.swapChainImageFormat = surfaceFormat.Format
	a.swapChainExtend = extend

	return nil
}

func (a *VulkanTutorialApp) recreateSwapChain() error {
	for true {
		width, height := a.window.GetFramebufferSize()
		if width != 0 || height != 0 {
			break
		}

		glfw.WaitEvents()
	}

	vk.DeviceWaitIdle(a.device)

	a.cleanupSwapChain()

	if err := a.createSwapChain(); err != nil {
		return fmt.Errorf("createSwapChain: %w", err)
	}
	if err := a.createImageViews(); err != nil {
		return fmt.Errorf("createImageViews: %w", err)
	}
	if err := a.createDepthResources(); err != nil {
		return fmt.Errorf("createDepthResources: %w", err)
	}
	if err := a.createFramebuffers(); err != nil {
		return fmt.Errorf("createFramebuffers: %w", err)
	}

	return nil
}

func (a *VulkanTutorialApp) createImageViews() error {
	for i, swapChainImage := range a.swapChainImages {
		swapChainImage := swapChainImage
		imageView, err := a.createImageView(
			swapChainImage,
			a.swapChainImageFormat,
			vk.ImageAspectFlags(vk.ImageAspectColorBit),
		)
		if err != nil {
			return fmt.Errorf("failed to create image %d: %w", i, err)
		}

		a.swapChainImageViews = append(a.swapChainImageViews, imageView)
	}

	return nil
}

func (a *VulkanTutorialApp) createRenderPass() error {
	depthFormat, err := a.findDepthFormat()
	if err != nil {
		return fmt.Errorf("cannot find suitable depth image format: %w", err)
	}

	depthAttachment := vk.AttachmentDescription{
		Format:         depthFormat,
		Samples:        vk.SampleCount1Bit,
		LoadOp:         vk.AttachmentLoadOpClear,
		StoreOp:        vk.AttachmentStoreOpDontCare,
		StencilLoadOp:  vk.AttachmentLoadOpDontCare,
		StencilStoreOp: vk.AttachmentStoreOpDontCare,
		InitialLayout:  vk.ImageLayoutUndefined,
		FinalLayout:    vk.ImageLayoutDepthStencilAttachmentOptimal,
	}

	depthAttachmentRef := vk.AttachmentReference{
		Attachment: 1,
		Layout:     vk.ImageLayoutDepthStencilAttachmentOptimal,
	}

	colorAttachment := vk.AttachmentDescription{
		Format:         a.swapChainImageFormat,
		Samples:        vk.SampleCount1Bit,
		LoadOp:         vk.AttachmentLoadOpClear,
		StoreOp:        vk.AttachmentStoreOpStore,
		StencilLoadOp:  vk.AttachmentLoadOpDontCare,
		StencilStoreOp: vk.AttachmentStoreOpDontCare,
		InitialLayout:  vk.ImageLayoutUndefined,
		FinalLayout:    vk.ImageLayoutPresentSrc,
	}

	colorAttachmentRef := vk.AttachmentReference{
		Attachment: 0,
		Layout:     vk.ImageLayoutColorAttachmentOptimal,
	}

	subpass := vk.SubpassDescription{
		PipelineBindPoint:       vk.PipelineBindPointGraphics,
		ColorAttachmentCount:    1,
		PColorAttachments:       []vk.AttachmentReference{colorAttachmentRef},
		PDepthStencilAttachment: &depthAttachmentRef,
	}

	dependency := vk.SubpassDependency{
		SrcSubpass: vk.SubpassExternal,
		DstSubpass: 0,
		SrcStageMask: vk.PipelineStageFlags(vk.PipelineStageColorAttachmentOutputBit) |
			vk.PipelineStageFlags(vk.PipelineStageEarlyFragmentTestsBit),
		SrcAccessMask: 0,
		DstStageMask: vk.PipelineStageFlags(vk.PipelineStageColorAttachmentOutputBit) |
			vk.PipelineStageFlags(vk.PipelineStageEarlyFragmentTestsBit),
		DstAccessMask: vk.AccessFlags(vk.AccessColorAttachmentWriteBit) |
			vk.AccessFlags(vk.AccessDepthStencilAttachmentWriteBit),
	}

	attachments := []vk.AttachmentDescription{
		colorAttachment,
		depthAttachment,
	}

	rederPassInfo := vk.RenderPassCreateInfo{
		SType:           vk.StructureTypeRenderPassCreateInfo,
		AttachmentCount: uint32(len(attachments)),
		PAttachments:    attachments,
		SubpassCount:    1,
		PSubpasses:      []vk.SubpassDescription{subpass},
		DependencyCount: 1,
		PDependencies:   []vk.SubpassDependency{dependency},
	}

	var renderPass vk.RenderPass
	res := vk.CreateRenderPass(a.device, &rederPassInfo, nil, &renderPass)
	if err := vk.Error(res); err != nil {
		return fmt.Errorf("failed to create render pass: %w", err)
	}
	a.renderPass = renderPass

	return nil
}

func (a *VulkanTutorialApp) createDescriptorSetLayout() error {
	uboLayoutBinding := vk.DescriptorSetLayoutBinding{
		Binding:            0,
		DescriptorType:     vk.DescriptorTypeUniformBuffer,
		DescriptorCount:    1,
		StageFlags:         vk.ShaderStageFlags(vk.ShaderStageVertexBit),
		PImmutableSamplers: nil,
	}

	samplerLayoutBinding := vk.DescriptorSetLayoutBinding{
		Binding:            1,
		DescriptorCount:    1,
		DescriptorType:     vk.DescriptorTypeCombinedImageSampler,
		PImmutableSamplers: nil,
		StageFlags:         vk.ShaderStageFlags(vk.ShaderStageFragmentBit),
	}

	bindings := []vk.DescriptorSetLayoutBinding{
		uboLayoutBinding,
		samplerLayoutBinding,
	}

	layoutInfo := vk.DescriptorSetLayoutCreateInfo{
		SType:        vk.StructureTypeDescriptorSetLayoutCreateInfo,
		BindingCount: uint32(len(bindings)),
		PBindings:    bindings,
	}

	var descriptorSetLayout vk.DescriptorSetLayout
	res := vk.CreateDescriptorSetLayout(a.device, &layoutInfo, nil, &descriptorSetLayout)
	if res != vk.Success {
		return fmt.Errorf("creating descriptor set layout: %w", vk.Error(res))
	}
	a.descriptorSetLayout = descriptorSetLayout

	return nil
}

func (a *VulkanTutorialApp) createGraphicsPipeline() error {
	vertShaderCode, err := shaders.FS.ReadFile("vert.spv")
	if err != nil {
		return fmt.Errorf("failed to read vertex shader bytecode: %w", err)
	}

	fragShaderCode, err := shaders.FS.ReadFile("frag.spv")
	if err != nil {
		return fmt.Errorf("failed to read fragment shader bytecode: %w", err)
	}

	if args.debug {
		log.Printf("vertex shader code size: %d", len(vertShaderCode))
		log.Printf("fragment shader code size: %d", len(fragShaderCode))
	}

	vertexShaderModule, err := a.createShaderModule(vertShaderCode)
	if err != nil {
		return fmt.Errorf("creating vertex shader module: %w", err)
	}
	defer vk.DestroyShaderModule(a.device, vertexShaderModule, nil)

	fragmentShaderModule, err := a.createShaderModule(fragShaderCode)
	if err != nil {
		return fmt.Errorf("creating fragment shader module: %w", err)
	}
	defer vk.DestroyShaderModule(a.device, fragmentShaderModule, nil)

	vertShaderStageInfo := vk.PipelineShaderStageCreateInfo{
		SType:  vk.StructureTypePipelineShaderStageCreateInfo,
		Stage:  vk.ShaderStageVertexBit,
		Module: vertexShaderModule,
		PName:  "main\x00",
	}

	fragShaderStageInfo := vk.PipelineShaderStageCreateInfo{
		SType:  vk.StructureTypePipelineShaderStageCreateInfo,
		Stage:  vk.ShaderStageFragmentBit,
		Module: fragmentShaderModule,
		PName:  "main\x00",
	}

	shaderStages := []vk.PipelineShaderStageCreateInfo{
		vertShaderStageInfo,
		fragShaderStageInfo,
	}

	bindingDescription := GetVertexBindingDescription()
	attributeDescriptions := GetVertexAttributeDescriptions()

	vertexInputInfo := vk.PipelineVertexInputStateCreateInfo{
		SType: vk.StructureTypePipelineVertexInputStateCreateInfo,

		VertexBindingDescriptionCount: 1,
		PVertexBindingDescriptions:    []vk.VertexInputBindingDescription{bindingDescription},

		VertexAttributeDescriptionCount: uint32(len(attributeDescriptions)),
		PVertexAttributeDescriptions:    attributeDescriptions[:],
	}

	inputAssembly := vk.PipelineInputAssemblyStateCreateInfo{
		SType:                  vk.StructureTypePipelineInputAssemblyStateCreateInfo,
		Topology:               vk.PrimitiveTopologyTriangleList,
		PrimitiveRestartEnable: vk.False,
	}

	viewport := vk.Viewport{
		X:        0,
		Y:        0,
		Width:    float32(a.swapChainExtend.Width),
		Height:   float32(a.swapChainExtend.Height),
		MinDepth: 0,
		MaxDepth: 1,
	}

	scissor := vk.Rect2D{
		Offset: vk.Offset2D{X: 0, Y: 0},
		Extent: a.swapChainExtend,
	}

	dynamicStates := []vk.DynamicState{
		vk.DynamicStateViewport,
		vk.DynamicStateScissor,
	}

	dynamicState := vk.PipelineDynamicStateCreateInfo{
		SType:             vk.StructureTypePipelineDynamicStateCreateInfo,
		DynamicStateCount: uint32(len(dynamicStates)),
		PDynamicStates:    dynamicStates,
	}

	viewportState := vk.PipelineViewportStateCreateInfo{
		SType:         vk.StructureTypePipelineViewportStateCreateInfo,
		ViewportCount: 1,
		ScissorCount:  1,
		PViewports:    []vk.Viewport{viewport},
		PScissors:     []vk.Rect2D{scissor},
	}

	rasterizer := vk.PipelineRasterizationStateCreateInfo{
		SType:                   vk.StructureTypePipelineRasterizationStateCreateInfo,
		DepthClampEnable:        vk.False,
		RasterizerDiscardEnable: vk.False,
		PolygonMode:             vk.PolygonModeFill,
		LineWidth:               1,
		CullMode:                vk.CullModeFlags(vk.CullModeBackBit),
		FrontFace:               vk.FrontFaceCounterClockwise,
		DepthBiasEnable:         vk.False,
	}

	multisampling := vk.PipelineMultisampleStateCreateInfo{
		SType:                 vk.StructureTypePipelineMultisampleStateCreateInfo,
		SampleShadingEnable:   vk.False,
		RasterizationSamples:  vk.SampleCount1Bit,
		MinSampleShading:      1,
		AlphaToCoverageEnable: vk.False,
		AlphaToOneEnable:      vk.False,
	}

	colorBlnedAttachment := vk.PipelineColorBlendAttachmentState{
		ColorWriteMask: vk.ColorComponentFlags(
			vk.ColorComponentRBit |
				vk.ColorComponentGBit |
				vk.ColorComponentBBit |
				vk.ColorComponentABit,
		),
		BlendEnable:         vk.False,
		SrcColorBlendFactor: vk.BlendFactorOne,
		DstColorBlendFactor: vk.BlendFactorZero,
		ColorBlendOp:        vk.BlendOpAdd,
		SrcAlphaBlendFactor: vk.BlendFactorOne,
		DstAlphaBlendFactor: vk.BlendFactorZero,
		AlphaBlendOp:        vk.BlendOpAdd,
	}

	colorBlending := vk.PipelineColorBlendStateCreateInfo{
		SType:           vk.StructureTypePipelineColorBlendStateCreateInfo,
		LogicOpEnable:   vk.False,
		LogicOp:         vk.LogicOpCopy,
		AttachmentCount: 1,
		PAttachments: []vk.PipelineColorBlendAttachmentState{
			colorBlnedAttachment,
		},
	}

	pipelineLayoutInfo := vk.PipelineLayoutCreateInfo{
		SType:          vk.StructureTypePipelineLayoutCreateInfo,
		SetLayoutCount: 1,
		PSetLayouts:    []vk.DescriptorSetLayout{a.descriptorSetLayout},
	}

	var pipelineLayout vk.PipelineLayout
	res := vk.CreatePipelineLayout(a.device, &pipelineLayoutInfo, nil, &pipelineLayout)
	if err := vk.Error(res); err != nil {
		return fmt.Errorf("failed to create pipeline layout: %w", err)
	}
	a.pipelineLayout = pipelineLayout

	depthStencil := vk.PipelineDepthStencilStateCreateInfo{
		SType:                 vk.StructureTypePipelineDepthStencilStateCreateInfo,
		DepthTestEnable:       vk.True,
		DepthWriteEnable:      vk.True,
		DepthCompareOp:        vk.CompareOpLess,
		DepthBoundsTestEnable: vk.False,
		MinDepthBounds:        0,
		MaxDepthBounds:        1,
		StencilTestEnable:     vk.False,
	}

	pipelineInfo := vk.GraphicsPipelineCreateInfo{
		SType:               vk.StructureTypeGraphicsPipelineCreateInfo,
		StageCount:          uint32(len(shaderStages)),
		PStages:             shaderStages,
		PVertexInputState:   &vertexInputInfo,
		PInputAssemblyState: &inputAssembly,
		PViewportState:      &viewportState,
		PRasterizationState: &rasterizer,
		PMultisampleState:   &multisampling,
		PDepthStencilState:  &depthStencil,
		PColorBlendState:    &colorBlending,
		PDynamicState:       &dynamicState,
		Layout:              a.pipelineLayout,
		RenderPass:          a.renderPass,
		Subpass:             0,
		BasePipelineHandle:  vk.Pipeline(vk.NullHandle),
		BasePipelineIndex:   -1,
	}

	pipelines := make([]vk.Pipeline, 1)
	res = vk.CreateGraphicsPipelines(
		a.device,
		vk.PipelineCache(vk.NullHandle),
		1,
		[]vk.GraphicsPipelineCreateInfo{pipelineInfo},
		nil,
		pipelines,
	)
	if err := vk.Error(res); err != nil {
		return fmt.Errorf("failed to create graphics pipeline: %w", err)
	}
	a.graphicsPipline = pipelines[0]

	return nil
}

func (a *VulkanTutorialApp) createFramebuffers() error {
	a.swapChainFramebuffers = make([]vk.Framebuffer, len(a.swapChainImageViews))

	for i, swapChainView := range a.swapChainImageViews {
		swapChainView := swapChainView

		attachments := []vk.ImageView{
			swapChainView,
			a.depthImageView,
		}

		frameBufferInfo := vk.FramebufferCreateInfo{
			SType:           vk.StructureTypeFramebufferCreateInfo,
			RenderPass:      a.renderPass,
			AttachmentCount: uint32(len(attachments)),
			PAttachments:    attachments,
			Width:           a.swapChainExtend.Width,
			Height:          a.swapChainExtend.Height,
			Layers:          1,
		}

		var frameBuffer vk.Framebuffer
		res := vk.CreateFramebuffer(a.device, &frameBufferInfo, nil, &frameBuffer)
		if err := vk.Error(res); err != nil {
			return fmt.Errorf("failed to create frame buffer %d: %w", i, err)
		}

		a.swapChainFramebuffers[i] = frameBuffer
	}

	return nil
}

func (a *VulkanTutorialApp) createCommandPool() error {
	queueFamilyIndices := a.findQueueFamilies(a.physicalDevice)
	poolInfo := vk.CommandPoolCreateInfo{
		SType: vk.StructureTypeCommandPoolCreateInfo,
		Flags: vk.CommandPoolCreateFlags(
			vk.CommandPoolCreateResetCommandBufferBit,
		),
		QueueFamilyIndex: queueFamilyIndices.Graphics.Get(),
	}

	var commandPool vk.CommandPool
	res := vk.CreateCommandPool(a.device, &poolInfo, nil, &commandPool)
	if err := vk.Error(res); err != nil {
		return fmt.Errorf("failed to create command pool: %w", err)
	}
	a.commandPool = commandPool

	return nil
}

func (a *VulkanTutorialApp) createVertexBuffer() error {
	bufferSize := vk.DeviceSize(uint32(len(a.vertices)) * GetVertexSize())

	// Create the staging buffer
	var (
		stagingBuffer       vk.Buffer
		stagingBufferMemory vk.DeviceMemory
	)
	err := a.createBuffer(
		bufferSize,
		vk.BufferUsageFlags(vk.BufferUsageTransferSrcBit),
		vk.MemoryPropertyFlags(vk.MemoryPropertyHostVisibleBit)|
			vk.MemoryPropertyFlags(vk.MemoryPropertyHostCoherentBit),
		&stagingBuffer,
		&stagingBufferMemory,
	)
	if err != nil {
		return fmt.Errorf("creating the staging buffer: %w", err)
	}

	defer func() {
		vk.DestroyBuffer(a.device, stagingBuffer, nil)
		vk.FreeMemory(a.device, stagingBufferMemory, nil)
	}()

	// Copy the data from host to staging buffer
	var pData unsafe.Pointer
	vk.MapMemory(a.device, stagingBufferMemory, 0, bufferSize, 0, &pData)

	bytesSlice := unsafer.SliceToBytes(a.vertices)

	vk.Memcopy(pData, bytesSlice)
	vk.UnmapMemory(a.device, stagingBufferMemory)

	// Create the device local buffer
	var (
		vertexBuffer       vk.Buffer
		vertexBufferMemory vk.DeviceMemory
	)

	err = a.createBuffer(
		bufferSize,
		vk.BufferUsageFlags(vk.BufferUsageTransferDstBit)|
			vk.BufferUsageFlags(vk.BufferUsageVertexBufferBit),
		vk.MemoryPropertyFlags(vk.MemoryPropertyDeviceLocalBit),
		&vertexBuffer,
		&vertexBufferMemory,
	)
	if err != nil {
		return fmt.Errorf("creating the vertex buffer: %w", err)
	}
	a.vertexBuffer = vertexBuffer
	a.vertexBufferMemory = vertexBufferMemory

	// Copy data from the staging buffer to the device local buffer which is our
	// vertex buffer
	if err := a.copyBuffer(stagingBuffer, a.vertexBuffer, bufferSize); err != nil {
		return fmt.Errorf("failed to copy staging buffer into vertex: %w", err)
	}

	return nil
}

func (a *VulkanTutorialApp) createIndexBuffer() error {
	bufferSize := vk.DeviceSize(
		uint32(len(a.indices)) * uint32(unsafe.Sizeof(a.indices[0])),
	)

	// Create the staging buffer
	var (
		stagingBuffer       vk.Buffer
		stagingBufferMemory vk.DeviceMemory
	)
	err := a.createBuffer(
		bufferSize,
		vk.BufferUsageFlags(vk.BufferUsageTransferSrcBit),
		vk.MemoryPropertyFlags(vk.MemoryPropertyHostVisibleBit)|
			vk.MemoryPropertyFlags(vk.MemoryPropertyHostCoherentBit),
		&stagingBuffer,
		&stagingBufferMemory,
	)
	if err != nil {
		return fmt.Errorf("creating the staging buffer: %w", err)
	}

	defer func() {
		vk.DestroyBuffer(a.device, stagingBuffer, nil)
		vk.FreeMemory(a.device, stagingBufferMemory, nil)
	}()

	// Copy the data from host to staging buffer
	var pData unsafe.Pointer
	vk.MapMemory(a.device, stagingBufferMemory, 0, bufferSize, 0, &pData)

	bytesSlice := unsafer.SliceToBytes(a.indices)

	vk.Memcopy(pData, bytesSlice)
	vk.UnmapMemory(a.device, stagingBufferMemory)

	// Create the device local buffer
	var (
		indexBuffer       vk.Buffer
		indexBufferMemory vk.DeviceMemory
	)

	err = a.createBuffer(
		bufferSize,
		vk.BufferUsageFlags(vk.BufferUsageTransferDstBit)|
			vk.BufferUsageFlags(vk.BufferUsageIndexBufferBit),
		vk.MemoryPropertyFlags(vk.MemoryPropertyDeviceLocalBit),
		&indexBuffer,
		&indexBufferMemory,
	)
	if err != nil {
		return fmt.Errorf("creating the index buffer: %w", err)
	}
	a.indexBuffer = indexBuffer
	a.indexBufferMemory = indexBufferMemory

	// Copy data from the staging buffer to the device local buffer which is our
	// index buffer
	if err := a.copyBuffer(stagingBuffer, a.indexBuffer, bufferSize); err != nil {
		return fmt.Errorf("failed to copy staging buffer into index: %w", err)
	}

	return nil
}

func (a *VulkanTutorialApp) copyBuffer(
	srcBuffer vk.Buffer,
	dstBuffer vk.Buffer,
	size vk.DeviceSize,
) error {
	commandBuffer, err := a.beginSingleTimeCommands()
	if err != nil {
		return fmt.Errorf("failed to begin single time commands: %w", err)
	}

	copyRegion := vk.BufferCopy{
		SrcOffset: 0,
		DstOffset: 0,
		Size:      size,
	}

	vk.CmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, []vk.BufferCopy{copyRegion})

	return a.endSingleTimeCommands(commandBuffer)
}

func (a *VulkanTutorialApp) beginSingleTimeCommands() (vk.CommandBuffer, error) {
	allocInfo := vk.CommandBufferAllocateInfo{
		SType:              vk.StructureTypeCommandBufferAllocateInfo,
		Level:              vk.CommandBufferLevelPrimary,
		CommandPool:        a.commandPool,
		CommandBufferCount: 1,
	}

	commandBuffers := make([]vk.CommandBuffer, 1)
	res := vk.AllocateCommandBuffers(
		a.device,
		&allocInfo,
		commandBuffers,
	)
	if res != vk.Success {
		return nil, fmt.Errorf("failed to allocate command buffer: %w", vk.Error(res))
	}
	commandBuffer := commandBuffers[0]

	beginInfo := vk.CommandBufferBeginInfo{
		SType: vk.StructureTypeCommandBufferBeginInfo,
		Flags: vk.CommandBufferUsageFlags(vk.CommandBufferUsageOneTimeSubmitBit),
	}

	vk.BeginCommandBuffer(commandBuffer, &beginInfo)

	return commandBuffer, nil
}

func (a *VulkanTutorialApp) endSingleTimeCommands(commandBuffer vk.CommandBuffer) error {
	commandBuffers := []vk.CommandBuffer{commandBuffer}

	defer func() {
		vk.FreeCommandBuffers(a.device, a.commandPool, 1, commandBuffers)
	}()

	res := vk.EndCommandBuffer(commandBuffer)
	if res != vk.Success {
		return fmt.Errorf("failed end command buffer: %w", vk.Error(res))
	}

	submitInfo := vk.SubmitInfo{
		SType:              vk.StructureTypeSubmitInfo,
		CommandBufferCount: 1,
		PCommandBuffers:    commandBuffers,
	}

	res = vk.QueueSubmit(a.graphicsQueue, 1, []vk.SubmitInfo{submitInfo}, vk.NullFence)
	if res != vk.Success {
		return fmt.Errorf("failed to submit to graphics queue: %w", vk.Error(res))
	}

	res = vk.QueueWaitIdle(a.graphicsQueue)
	if res != vk.Success {
		return fmt.Errorf("failed to wait on graphics queue idle: %w", vk.Error(res))
	}

	return nil
}

func (a *VulkanTutorialApp) transitionImageLayout(
	image vk.Image,
	format vk.Format,
	oldLayout vk.ImageLayout,
	newLayout vk.ImageLayout,
) error {
	commandBuffer, err := a.beginSingleTimeCommands()
	if err != nil {
		return fmt.Errorf("failed to begin single time commands: %w", err)
	}

	barrier := vk.ImageMemoryBarrier{
		SType:               vk.StructureTypeImageMemoryBarrier,
		OldLayout:           oldLayout,
		NewLayout:           newLayout,
		SrcQueueFamilyIndex: vk.QueueFamilyIgnored,
		DstQueueFamilyIndex: vk.QueueFamilyIgnored,
		Image:               image,
		SubresourceRange: vk.ImageSubresourceRange{
			AspectMask:     vk.ImageAspectFlags(vk.ImageAspectColorBit),
			BaseMipLevel:   0,
			LevelCount:     1,
			BaseArrayLayer: 0,
			LayerCount:     1,
		},
		SrcAccessMask: 0,
		DstAccessMask: 0,
	}

	var (
		sourceStage      vk.PipelineStageFlags
		destinationStage vk.PipelineStageFlags
	)

	if oldLayout == vk.ImageLayoutUndefined &&
		newLayout == vk.ImageLayoutTransferDstOptimal {

		barrier.SrcAccessMask = 0
		barrier.DstAccessMask = vk.AccessFlags(vk.AccessTransferWriteBit)

		sourceStage = vk.PipelineStageFlags(vk.PipelineStageTopOfPipeBit)
		destinationStage = vk.PipelineStageFlags(vk.PipelineStageTransferBit)

	} else if oldLayout == vk.ImageLayoutTransferDstOptimal &&
		newLayout == vk.ImageLayoutShaderReadOnlyOptimal {

		barrier.SrcAccessMask = vk.AccessFlags(vk.AccessTransferWriteBit)
		barrier.DstAccessMask = vk.AccessFlags(vk.AccessShaderReadBit)

		sourceStage = vk.PipelineStageFlags(vk.PipelineStageTransferBit)
		destinationStage = vk.PipelineStageFlags(vk.PipelineStageFragmentShaderBit)

	} else {
		return fmt.Errorf("unsupported layout transition")
	}

	vk.CmdPipelineBarrier(
		commandBuffer,
		sourceStage, destinationStage,
		0,
		0, nil,
		0, nil,
		1, []vk.ImageMemoryBarrier{barrier},
	)

	return a.endSingleTimeCommands(commandBuffer)
}

func (a *VulkanTutorialApp) copyBufferToImage(
	buffer vk.Buffer,
	image vk.Image,
	width, height uint32,
) error {
	commandBuffer, err := a.beginSingleTimeCommands()
	if err != nil {
		return fmt.Errorf("failed to beging single time command buffer: %w", err)
	}

	region := vk.BufferImageCopy{
		BufferOffset:      0,
		BufferRowLength:   0,
		BufferImageHeight: 0,

		ImageSubresource: vk.ImageSubresourceLayers{
			AspectMask:     vk.ImageAspectFlags(vk.ImageAspectColorBit),
			MipLevel:       0,
			BaseArrayLayer: 0,
			LayerCount:     1,
		},

		ImageOffset: vk.Offset3D{
			X: 0, Y: 0, Z: 0,
		},

		ImageExtent: vk.Extent3D{
			Width:  width,
			Height: height,
			Depth:  1,
		},
	}

	vk.CmdCopyBufferToImage(
		commandBuffer,
		buffer,
		image,
		vk.ImageLayoutTransferDstOptimal,
		1,
		[]vk.BufferImageCopy{region},
	)

	return a.endSingleTimeCommands(commandBuffer)
}

func (a *VulkanTutorialApp) createUniformBuffers() error {
	bufferSize := vk.DeviceSize(unsafe.Sizeof(UniformBufferObject{}))

	for i := 0; i < maxFramesInFlight; i++ {
		var (
			buffer       vk.Buffer
			bufferMemory vk.DeviceMemory
		)
		err := a.createBuffer(
			bufferSize,
			vk.BufferUsageFlags(vk.BufferUsageUniformBufferBit),
			vk.MemoryPropertyFlags(vk.MemoryPropertyHostVisibleBit)|
				vk.MemoryPropertyFlags(vk.MemoryPropertyHostCoherentBit),
			&buffer,
			&bufferMemory,
		)
		if err != nil {
			return fmt.Errorf("creating buffer[%d]: %w", i, err)
		}

		a.uniformBuffers = append(a.uniformBuffers, buffer)
		a.uniformBuffersMemory = append(a.uniformBuffersMemory, bufferMemory)

		var pData unsafe.Pointer
		vk.MapMemory(a.device, a.uniformBuffersMemory[i], 0, bufferSize, 0, &pData)
		a.uniformBuffersMapped = append(a.uniformBuffersMapped, pData)
	}

	return nil
}

func (a *VulkanTutorialApp) createDescriptorPool() error {
	poolSizes := []vk.DescriptorPoolSize{
		{
			Type:            vk.DescriptorTypeUniformBuffer,
			DescriptorCount: maxFramesInFlight,
		},
		{
			Type:            vk.DescriptorTypeCombinedImageSampler,
			DescriptorCount: maxFramesInFlight,
		},
	}

	poolInfo := vk.DescriptorPoolCreateInfo{
		SType:         vk.StructureTypeDescriptorPoolCreateInfo,
		PoolSizeCount: uint32(len(poolSizes)),
		PPoolSizes:    poolSizes,
		MaxSets:       maxFramesInFlight,
	}

	var descriptorPool vk.DescriptorPool
	res := vk.CreateDescriptorPool(a.device, &poolInfo, nil, &descriptorPool)
	if res != vk.Success {
		return fmt.Errorf("failed to create descriptor pool: %w", vk.Error(res))
	}
	a.descriptorPool = descriptorPool

	return nil
}

func (a *VulkanTutorialApp) createDescriptorSets() error {
	layouts := []vk.DescriptorSetLayout{
		a.descriptorSetLayout,
		a.descriptorSetLayout,
	}

	allocInfo := vk.DescriptorSetAllocateInfo{
		SType:              vk.StructureTypeDescriptorSetAllocateInfo,
		DescriptorPool:     a.descriptorPool,
		DescriptorSetCount: maxFramesInFlight,
		PSetLayouts:        layouts,
	}

	a.descriptorSets = make([]vk.DescriptorSet, maxFramesInFlight)

	res := vk.AllocateDescriptorSets(a.device, &allocInfo, &a.descriptorSets[0])
	if res != vk.Success {
		return fmt.Errorf("failed to allocate descriptor set: %w", vk.Error(res))
	}

	for i := 0; i < maxFramesInFlight; i++ {
		bufferInfo := vk.DescriptorBufferInfo{
			Buffer: a.uniformBuffers[i],
			Offset: 0,
			Range:  vk.DeviceSize(vk.WholeSize),
		}

		imageInfo := vk.DescriptorImageInfo{
			ImageLayout: vk.ImageLayoutShaderReadOnlyOptimal,
			ImageView:   a.textureImageView,
			Sampler:     a.textureSampler,
		}

		descriptorWrites := []vk.WriteDescriptorSet{
			{
				SType:           vk.StructureTypeWriteDescriptorSet,
				DstSet:          a.descriptorSets[i],
				DstBinding:      0,
				DstArrayElement: 0,
				DescriptorType:  vk.DescriptorTypeUniformBuffer,
				DescriptorCount: 1,
				PBufferInfo:     []vk.DescriptorBufferInfo{bufferInfo},
			},
			{
				SType:           vk.StructureTypeWriteDescriptorSet,
				DstSet:          a.descriptorSets[i],
				DstBinding:      1,
				DstArrayElement: 0,
				DescriptorType:  vk.DescriptorTypeCombinedImageSampler,
				DescriptorCount: 1,
				PImageInfo:      []vk.DescriptorImageInfo{imageInfo},
			},
		}

		vk.UpdateDescriptorSets(
			a.device,
			uint32(len(descriptorWrites)),
			descriptorWrites,
			0,
			nil,
		)
	}

	return nil
}

func (a *VulkanTutorialApp) createBuffer(
	size vk.DeviceSize,
	usage vk.BufferUsageFlags,
	properties vk.MemoryPropertyFlags,
	buffer *vk.Buffer,
	bufferMemory *vk.DeviceMemory,
) error {
	bufferInfo := vk.BufferCreateInfo{
		SType:       vk.StructureTypeBufferCreateInfo,
		Size:        size,
		Usage:       usage,
		SharingMode: vk.SharingModeExclusive,
	}

	res := vk.CreateBuffer(a.device, &bufferInfo, nil, buffer)
	if res != vk.Success {
		return fmt.Errorf("failed to create vertex buffer: %w", vk.Error(res))
	}

	var memRequirements vk.MemoryRequirements
	vk.GetBufferMemoryRequirements(a.device, *buffer, &memRequirements)
	memRequirements.Deref()

	memTypeIndex, err := a.findMemoryType(memRequirements.MemoryTypeBits, properties)
	if err != nil {
		return err
	}

	allocInfo := vk.MemoryAllocateInfo{
		SType:           vk.StructureTypeMemoryAllocateInfo,
		AllocationSize:  memRequirements.Size,
		MemoryTypeIndex: memTypeIndex,
	}

	res = vk.AllocateMemory(a.device, &allocInfo, nil, bufferMemory)
	if res != vk.Success {
		return fmt.Errorf("failed to allocate vertex buffer memory: %s", vk.Error(res))
	}

	res = vk.BindBufferMemory(a.device, *buffer, *bufferMemory, 0)
	if res != vk.Success {
		return fmt.Errorf("failed to bind buffer memory: %w", vk.Error(res))
	}

	return nil
}

func (a *VulkanTutorialApp) createImage(
	width uint32,
	height uint32,
	format vk.Format,
	tiling vk.ImageTiling,
	usage vk.ImageUsageFlags,
	properties vk.MemoryPropertyFlags,
	image *vk.Image,
	imageMemory *vk.DeviceMemory,
) error {
	imageInfo := vk.ImageCreateInfo{
		SType:     vk.StructureTypeImageCreateInfo,
		ImageType: vk.ImageType2d,
		Extent: vk.Extent3D{
			Width:  width,
			Height: height,
			Depth:  1,
		},
		MipLevels:     1,
		ArrayLayers:   1,
		Format:        format,
		Tiling:        tiling,
		InitialLayout: vk.ImageLayoutUndefined,
		Usage:         usage,
		SharingMode:   vk.SharingModeExclusive,
		Samples:       vk.SampleCount1Bit,
	}

	res := vk.CreateImage(a.device, &imageInfo, nil, image)
	if res != vk.Success {
		return fmt.Errorf("failed to create an image: %w", vk.Error(res))
	}

	var memRequirements vk.MemoryRequirements
	vk.GetImageMemoryRequirements(a.device, *image, &memRequirements)
	memRequirements.Deref()

	memTypeIndex, err := a.findMemoryType(memRequirements.MemoryTypeBits, properties)
	if err != nil {
		return err
	}

	allocInfo := vk.MemoryAllocateInfo{
		SType:           vk.StructureTypeMemoryAllocateInfo,
		AllocationSize:  memRequirements.Size,
		MemoryTypeIndex: memTypeIndex,
	}

	res = vk.AllocateMemory(a.device, &allocInfo, nil, imageMemory)
	if res != vk.Success {
		return fmt.Errorf("failed to allocate image buffer memory: %s", vk.Error(res))
	}

	res = vk.BindImageMemory(a.device, *image, *imageMemory, 0)
	if res != vk.Success {
		return fmt.Errorf("failed to bind image memory: %w", vk.Error(res))
	}

	return nil
}

func (a *VulkanTutorialApp) findMemoryType(
	typeFilter uint32,
	properties vk.MemoryPropertyFlags,
) (uint32, error) {
	var memProperties vk.PhysicalDeviceMemoryProperties
	vk.GetPhysicalDeviceMemoryProperties(a.physicalDevice, &memProperties)
	memProperties.Deref()

	for i := uint32(0); i < memProperties.MemoryTypeCount; i++ {
		memType := memProperties.MemoryTypes[i]
		memType.Deref()

		if typeFilter&(1<<i) == 0 {
			continue
		}

		if memType.PropertyFlags&properties != properties {
			continue
		}

		return i, nil
	}

	return 0, fmt.Errorf("failed to find suitable memory type")
}

func (a *VulkanTutorialApp) createDepthResources() error {
	depthFormat, err := a.findDepthFormat()
	if err != nil {
		return fmt.Errorf("could not find suitable depth image format: %w", err)
	}

	var (
		depthImage       vk.Image
		depthImageMemory vk.DeviceMemory
	)

	err = a.createImage(
		a.swapChainExtend.Width,
		a.swapChainExtend.Height,
		depthFormat,
		vk.ImageTilingOptimal,
		vk.ImageUsageFlags(vk.ImageUsageDepthStencilAttachmentBit),
		vk.MemoryPropertyFlags(vk.MemoryPropertyDeviceLocalBit),
		&depthImage,
		&depthImageMemory,
	)
	if err != nil {
		return fmt.Errorf("could not create depth image: %w", err)
	}

	a.depthImage = depthImage
	a.depthImageMemory = depthImageMemory

	depthImageView, err := a.createImageView(
		depthImage,
		depthFormat,
		vk.ImageAspectFlags(vk.ImageAspectDepthBit),
	)
	if err != nil {
		return fmt.Errorf("failed to create depth image view: %w", err)
	}
	a.depthImageView = depthImageView

	return nil
}

func (a *VulkanTutorialApp) findSupportedFormat(
	candidates []vk.Format,
	tiling vk.ImageTiling,
	features vk.FormatFeatureFlags,
) (vk.Format, error) {
	for _, format := range candidates {
		var props vk.FormatProperties
		vk.GetPhysicalDeviceFormatProperties(a.physicalDevice, format, &props)
		props.Deref()

		if tiling == vk.ImageTilingLinear &&
			(props.LinearTilingFeatures&features) == features {
			return format, nil
		}

		if tiling == vk.ImageTilingOptimal &&
			(props.OptimalTilingFeatures&features) == features {
			return format, nil
		}

	}

	return 0, fmt.Errorf("could not find suitable format")
}

func (a *VulkanTutorialApp) findDepthFormat() (vk.Format, error) {
	return a.findSupportedFormat(
		[]vk.Format{
			vk.FormatD32Sfloat,
			vk.FormatD32SfloatS8Uint,
			vk.FormatD24UnormS8Uint,
		},
		vk.ImageTilingOptimal,
		vk.FormatFeatureFlags(vk.FormatFeatureDepthStencilAttachmentBit),
	)
}

func (a *VulkanTutorialApp) hasStencilComponent(format vk.Format) bool {
	return format == vk.FormatD32SfloatS8Uint || format == vk.FormatD24UnormS8Uint
}

func (a *VulkanTutorialApp) createTextureImage() error {
	fh, err := textures.FS.Open(a.texturePath)
	if err != nil {
		return fmt.Errorf("failed to open texture file: %w", err)
	}
	defer fh.Close()

	img, _, err := image.Decode(fh)
	if err != nil {
		return fmt.Errorf("failed to decode texture image: %w", err)
	}

	imgBoundsSize := img.Bounds().Size()
	texWidth := uint32(imgBoundsSize.X)
	texHeight := uint32(imgBoundsSize.Y)

	imgSize := vk.DeviceSize(texWidth * texHeight * 4)

	var (
		staginbBuffer       vk.Buffer
		stagingBufferMemory vk.DeviceMemory
	)

	err = a.createBuffer(
		imgSize,
		vk.BufferUsageFlags(vk.BufferUsageTransferSrcBit),
		vk.MemoryPropertyFlags(vk.MemoryPropertyHostVisibleBit)|
			vk.MemoryPropertyFlags(vk.MemoryPropertyHostCoherentBit),
		&staginbBuffer,
		&stagingBufferMemory,
	)
	if err != nil {
		return fmt.Errorf("failed to create texture GPU buffer: %w", err)
	}

	defer func() {
		vk.DestroyBuffer(a.device, staginbBuffer, nil)
		vk.FreeMemory(a.device, stagingBufferMemory, nil)
	}()

	var pData unsafe.Pointer
	vk.MapMemory(a.device, stagingBufferMemory, 0, imgSize, 0, &pData)
	defer vk.UnmapMemory(a.device, stagingBufferMemory)

	// convert the image to RGBA if it is not already
	b := img.Bounds()
	rgbaImg := image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	draw.Draw(rgbaImg, rgbaImg.Bounds(), img, b.Min, draw.Src)

	// copy its data
	vk.Memcopy(pData, rgbaImg.Pix)

	var (
		textureImage       vk.Image
		textureImageMemory vk.DeviceMemory
	)

	err = a.createImage(
		texWidth,
		texHeight,
		vk.FormatR8g8b8a8Srgb,
		vk.ImageTilingOptimal,
		vk.ImageUsageFlags(vk.ImageUsageTransferDstBit)|
			vk.ImageUsageFlags(vk.ImageUsageSampledBit),
		vk.MemoryPropertyFlags(vk.MemoryPropertyDeviceLocalBit),
		&textureImage,
		&textureImageMemory,
	)
	if err != nil {
		return fmt.Errorf("filed to create Vulkan image: %w", err)
	}
	a.textureImage = textureImage
	a.textureImageMemory = textureImageMemory

	err = a.transitionImageLayout(
		a.textureImage,
		vk.FormatR8g8b8a8Srgb,
		vk.ImageLayoutUndefined,
		vk.ImageLayoutTransferDstOptimal,
	)
	if err != nil {
		return fmt.Errorf("transition image layout: %w", err)
	}

	err = a.copyBufferToImage(staginbBuffer, a.textureImage, texWidth, texHeight)
	if err != nil {
		return fmt.Errorf("copying buffer to image: %w", err)
	}

	err = a.transitionImageLayout(
		a.textureImage,
		vk.FormatR8g8b8a8Srgb,
		vk.ImageLayoutTransferDstOptimal,
		vk.ImageLayoutShaderReadOnlyOptimal,
	)
	if err != nil {
		return fmt.Errorf("transitioning to read only optimal layout: %w", err)
	}

	return nil
}

func (a *VulkanTutorialApp) createTextureImageView() error {
	textureImageView, err := a.createImageView(
		a.textureImage,
		vk.FormatR8g8b8a8Srgb,
		vk.ImageAspectFlags(vk.ImageAspectColorBit),
	)
	if err != nil {
		return err
	}
	a.textureImageView = textureImageView

	return nil
}

func (a *VulkanTutorialApp) createImageView(
	image vk.Image,
	format vk.Format,
	aspectFlags vk.ImageAspectFlags,
) (vk.ImageView, error) {
	createInfo := vk.ImageViewCreateInfo{
		SType:    vk.StructureTypeImageViewCreateInfo,
		Image:    image,
		ViewType: vk.ImageViewType2d,
		Format:   format,
		Components: vk.ComponentMapping{
			R: vk.ComponentSwizzleIdentity,
			G: vk.ComponentSwizzleIdentity,
			B: vk.ComponentSwizzleIdentity,
			A: vk.ComponentSwizzleIdentity,
		},
		SubresourceRange: vk.ImageSubresourceRange{
			AspectMask:     aspectFlags,
			BaseMipLevel:   0,
			LevelCount:     1,
			BaseArrayLayer: 0,
			LayerCount:     1,
		},
	}

	var imageView vk.ImageView
	res := vk.CreateImageView(a.device, &createInfo, nil, &imageView)
	if err := vk.Error(res); err != nil {
		return nil, fmt.Errorf("failed to create image view: %w", err)
	}

	return imageView, nil
}

func (a *VulkanTutorialApp) createTextureSampler() error {
	var properties vk.PhysicalDeviceProperties
	vk.GetPhysicalDeviceProperties(a.physicalDevice, &properties)
	properties.Deref()
	properties.Limits.Deref()

	samplerInfo := vk.SamplerCreateInfo{
		SType:                   vk.StructureTypeSamplerCreateInfo,
		MagFilter:               vk.FilterLinear,
		MinFilter:               vk.FilterLinear,
		AddressModeU:            vk.SamplerAddressModeRepeat,
		AddressModeV:            vk.SamplerAddressModeRepeat,
		AddressModeW:            vk.SamplerAddressModeRepeat,
		AnisotropyEnable:        vk.True,
		MaxAnisotropy:           properties.Limits.MaxSamplerAnisotropy,
		UnnormalizedCoordinates: vk.False,
		CompareEnable:           vk.False,
		CompareOp:               vk.CompareOpAlways,
		MipmapMode:              vk.SamplerMipmapModeLinear,
		MipLodBias:              0,
		MinLod:                  0,
		MaxLod:                  0,
	}

	var sampler vk.Sampler
	res := vk.CreateSampler(a.device, &samplerInfo, nil, &sampler)
	if res != vk.Success {
		return fmt.Errorf("failed to create sampler: %w", vk.Error(res))
	}
	a.textureSampler = sampler

	return nil
}

func (a *VulkanTutorialApp) createCommandBuffer() error {
	allocInfo := vk.CommandBufferAllocateInfo{
		SType:              vk.StructureTypeCommandBufferAllocateInfo,
		CommandPool:        a.commandPool,
		Level:              vk.CommandBufferLevelPrimary,
		CommandBufferCount: 2,
	}

	commandBuffers := make([]vk.CommandBuffer, 2)
	res := vk.AllocateCommandBuffers(a.device, &allocInfo, commandBuffers)
	if err := vk.Error(res); err != nil {
		return fmt.Errorf("failed to allocate command buffer: %w", err)
	}
	a.commandBuffers = commandBuffers

	return nil
}

func (a *VulkanTutorialApp) recordCommandBuffer(
	commandBuffer vk.CommandBuffer,
	imageIndex uint32,
) error {
	beginInfo := vk.CommandBufferBeginInfo{
		SType: vk.StructureTypeCommandBufferBeginInfo,
		Flags: 0,
	}

	res := vk.BeginCommandBuffer(commandBuffer, &beginInfo)
	if err := vk.Error(res); err != nil {
		return fmt.Errorf("cannot add begin command to the buffer: %w", err)
	}

	var clearValues [2]vk.ClearValue

	clearValues[0].SetColor([]float32{0, 0, 0, 1})
	clearValues[1].SetDepthStencil(1, 0)

	renderPassInfo := vk.RenderPassBeginInfo{
		SType:       vk.StructureTypeRenderPassBeginInfo,
		RenderPass:  a.renderPass,
		Framebuffer: a.swapChainFramebuffers[imageIndex],
		RenderArea: vk.Rect2D{
			Offset: vk.Offset2D{
				X: 0,
				Y: 0,
			},
			Extent: a.swapChainExtend,
		},
		ClearValueCount: uint32(len(clearValues)),
		PClearValues:    clearValues[:],
	}

	vk.CmdBeginRenderPass(commandBuffer, &renderPassInfo, vk.SubpassContentsInline)
	vk.CmdBindPipeline(commandBuffer, vk.PipelineBindPointGraphics, a.graphicsPipline)

	vertexBuffers := []vk.Buffer{a.vertexBuffer}
	offsets := []vk.DeviceSize{0}
	vk.CmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets)

	vk.CmdBindIndexBuffer(commandBuffer, a.indexBuffer, 0, vk.IndexTypeUint16)

	viewport := vk.Viewport{
		X: 0, Y: 0,
		Width:    float32(a.swapChainExtend.Width),
		Height:   float32(a.swapChainExtend.Height),
		MinDepth: 0,
		MaxDepth: 1,
	}
	vk.CmdSetViewport(commandBuffer, 0, 1, []vk.Viewport{viewport})

	scissor := vk.Rect2D{
		Offset: vk.Offset2D{X: 0, Y: 0},
		Extent: a.swapChainExtend,
	}
	vk.CmdSetScissor(commandBuffer, 0, 1, []vk.Rect2D{scissor})

	vk.CmdBindDescriptorSets(
		commandBuffer,
		vk.PipelineBindPointGraphics,
		a.pipelineLayout,
		0,
		1,
		[]vk.DescriptorSet{a.descriptorSets[a.curentFrame]},
		0,
		nil,
	)

	vk.CmdDrawIndexed(commandBuffer, uint32(len(a.indices)), 1, 0, 0, 0)
	vk.CmdEndRenderPass(commandBuffer)

	if err := vk.Error(vk.EndCommandBuffer(commandBuffer)); err != nil {
		return fmt.Errorf("recording commands to buffer failed: %w", err)
	}
	return nil
}

func (a *VulkanTutorialApp) createSyncObjects() error {
	semaphoreInfo := vk.SemaphoreCreateInfo{
		SType: vk.StructureTypeSemaphoreCreateInfo,
	}

	fenceInfo := vk.FenceCreateInfo{
		SType: vk.StructureTypeFenceCreateInfo,
		Flags: vk.FenceCreateFlags(vk.FenceCreateSignaledBit),
	}

	for i := 0; i < maxFramesInFlight; i++ {
		var imageAvailabmeSem vk.Semaphore
		if err := vk.Error(
			vk.CreateSemaphore(a.device, &semaphoreInfo, nil, &imageAvailabmeSem),
		); err != nil {
			return fmt.Errorf("failed to create imageAvailabmeSem: %w", err)
		}
		a.imageAvailabmeSems = append(a.imageAvailabmeSems, imageAvailabmeSem)

		var renderFinishedSem vk.Semaphore
		if err := vk.Error(
			vk.CreateSemaphore(a.device, &semaphoreInfo, nil, &renderFinishedSem),
		); err != nil {
			return fmt.Errorf("failed to create renderFinishedSem: %w", err)
		}
		a.renderFinishedSems = append(a.renderFinishedSems, renderFinishedSem)

		var fence vk.Fence
		if err := vk.Error(
			vk.CreateFence(a.device, &fenceInfo, nil, &fence),
		); err != nil {
			return fmt.Errorf("failed to create inFlightFence: %w", err)
		}
		a.inFlightFences = append(a.inFlightFences, fence)
	}

	return nil
}

func (a *VulkanTutorialApp) createInsance() error {
	if a.enableValidationLayers && !a.checkValidationSupport() {
		return fmt.Errorf("validation layers requested but not available")
	}

	appInfo := vk.ApplicationInfo{
		SType:              vk.StructureTypeApplicationInfo,
		PApplicationName:   title + "\x00",
		ApplicationVersion: vk.MakeVersion(1, 0, 0),
		PEngineName:        "No Engine\x00",
		EngineVersion:      vk.MakeVersion(1, 0, 0),
		ApiVersion:         vk.ApiVersion10,
	}

	glfwExtensions := glfw.GetCurrentContext().GetRequiredInstanceExtensions()
	createInfo := vk.InstanceCreateInfo{
		SType:                   vk.StructureTypeInstanceCreateInfo,
		PApplicationInfo:        &appInfo,
		EnabledExtensionCount:   uint32(len(glfwExtensions)),
		PpEnabledExtensionNames: glfwExtensions,
	}

	if a.enableValidationLayers {
		createInfo.EnabledLayerCount = uint32(len(a.validationLayers))
		createInfo.PpEnabledLayerNames = a.validationLayers
	}

	var instance vk.Instance
	if err := vk.Error(vk.CreateInstance(&createInfo, nil, &instance)); err != nil {
		return fmt.Errorf("failed to create Vulkan instance: %w", err)
	}

	a.instance = instance
	return nil
}

// findQueueFamilies returns a FamilyIndeces populated with Vulkan queue families needed
// by the program.
func (a *VulkanTutorialApp) findQueueFamilies(
	device vk.PhysicalDevice,
) QueueFamilyIndices {
	indices := QueueFamilyIndices{}

	var queueFamilyCount uint32
	vk.GetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nil)

	queueFamilies := make([]vk.QueueFamilyProperties, queueFamilyCount)
	vk.GetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies)

	for i, family := range queueFamilies {
		family.Deref()

		if family.QueueFlags&vk.QueueFlags(vk.QueueGraphicsBit) != 0 {
			indices.Graphics.Set(uint32(i))
		}

		var hasPresent vk.Bool32
		err := vk.Error(
			vk.GetPhysicalDeviceSurfaceSupport(device, uint32(i), a.surface, &hasPresent),
		)
		if err != nil {
			log.Printf("error querying surface support for queue family %d: %s", i, err)
		} else if hasPresent.B() {
			indices.Present.Set(uint32(i))
		}

		if indices.IsComplete() {
			break
		}
	}

	return indices
}

func (a *VulkanTutorialApp) querySwapChainSupport(
	device vk.PhysicalDevice,
) swapChainSupportDetails {
	details := swapChainSupportDetails{}

	var capabilities vk.SurfaceCapabilities
	res := vk.GetPhysicalDeviceSurfaceCapabilities(device, a.surface, &capabilities)
	if err := vk.Error(res); err != nil {
		panic(fmt.Sprintf("failed to query device surface capabilities: %s", err))
	}
	capabilities.Deref()
	capabilities.CurrentExtent.Deref()
	capabilities.MinImageExtent.Deref()
	capabilities.MaxImageExtent.Deref()

	details.capabilities = capabilities

	var formatCount uint32
	res = vk.GetPhysicalDeviceSurfaceFormats(device, a.surface, &formatCount, nil)
	if err := vk.Error(res); err != nil {
		panic(fmt.Sprintf("failed to query device surface formats: %s", err))
	}

	if formatCount != 0 {
		formats := make([]vk.SurfaceFormat, formatCount)
		vk.GetPhysicalDeviceSurfaceFormats(device, a.surface, &formatCount, formats)
		for _, format := range formats {
			format.Deref()
			details.formats = append(details.formats, format)
		}
	}

	var presentModeCount uint32
	res = vk.GetPhysicalDeviceSurfacePresentModes(
		device, a.surface, &presentModeCount, nil,
	)
	if err := vk.Error(res); err != nil {
		panic(fmt.Sprintf("failed to query device surface present modes: %s", err))
	}

	if presentModeCount != 0 {
		presentModes := make([]vk.PresentMode, presentModeCount)
		vk.GetPhysicalDeviceSurfacePresentModes(
			device, a.surface, &presentModeCount, presentModes,
		)
		details.presentModes = presentModes
	}

	return details
}

// getDeviceScore returns how suitable is this device for the current program.
// Bigger score means better. Zero or negative means the device cannot be used.
func (a *VulkanTutorialApp) getDeviceScore(device vk.PhysicalDevice) uint32 {
	var (
		deviceScore uint32
		properties  vk.PhysicalDeviceProperties
	)

	vk.GetPhysicalDeviceProperties(device, &properties)
	properties.Deref()

	if properties.DeviceType == vk.PhysicalDeviceTypeDiscreteGpu {
		deviceScore += 1000
	} else {
		deviceScore++
	}

	if !a.isDeviceSuitable(device) {
		deviceScore = 0
	}

	if args.debug {
		log.Printf(
			"Available device: %s (score: %d)",
			vk.ToString(properties.DeviceName[:]),
			deviceScore,
		)
	}

	return deviceScore
}

func (a *VulkanTutorialApp) isDeviceSuitable(device vk.PhysicalDevice) bool {
	indices := a.findQueueFamilies(device)
	extensionsSupported := a.checkDeviceExtensionSupport(device)

	swapChainAdequate := false
	if extensionsSupported {
		swapChainSupport := a.querySwapChainSupport(device)
		swapChainAdequate = len(swapChainSupport.formats) > 0 &&
			len(swapChainSupport.presentModes) > 0
	}

	var supportedFeatures vk.PhysicalDeviceFeatures
	vk.GetPhysicalDeviceFeatures(device, &supportedFeatures)
	supportedFeatures.Deref()

	return indices.IsComplete() && extensionsSupported && swapChainAdequate &&
		supportedFeatures.SamplerAnisotropy.B()
}

func (a *VulkanTutorialApp) chooseSwapSurfaceFormat(
	availableFormats []vk.SurfaceFormat,
) vk.SurfaceFormat {
	for _, format := range availableFormats {
		if format.Format == vk.FormatB8g8r8a8Srgb &&
			format.ColorSpace == vk.ColorSpaceSrgbNonlinear {
			return format
		}
	}

	return availableFormats[0]
}

func (a *VulkanTutorialApp) chooseSwapPresentMode(
	available []vk.PresentMode,
) vk.PresentMode {
	for _, mode := range available {
		if mode == vk.PresentModeMailbox {
			return mode
		}
	}

	return vk.PresentModeFifo
}

func (a *VulkanTutorialApp) chooseSwapExtend(
	capabilities vk.SurfaceCapabilities,
) vk.Extent2D {
	if capabilities.CurrentExtent.Width != math.MaxUint32 {
		return capabilities.CurrentExtent
	}

	width, height := a.window.GetFramebufferSize()

	actualExtend := vk.Extent2D{
		Width:  uint32(width),
		Height: uint32(height),
	}

	actualExtend.Width = clamp(
		actualExtend.Width,
		capabilities.MinImageExtent.Width,
		capabilities.MaxImageExtent.Width,
	)

	actualExtend.Height = clamp(
		actualExtend.Height,
		capabilities.MinImageExtent.Height,
		capabilities.MaxImageExtent.Height,
	)

	fmt.Printf("actualExtend: %#v", actualExtend)

	return actualExtend
}

func (a *VulkanTutorialApp) checkDeviceExtensionSupport(device vk.PhysicalDevice) bool {
	var extensionsCount uint32
	res := vk.EnumerateDeviceExtensionProperties(device, "", &extensionsCount, nil)
	if err := vk.Error(res); err != nil {
		log.Printf(
			"WARNING: enumerating device (%d) extension properties count: %s",
			device,
			err,
		)
		return false
	}

	availableExtensions := make([]vk.ExtensionProperties, extensionsCount)
	res = vk.EnumerateDeviceExtensionProperties(device, "", &extensionsCount,
		availableExtensions)
	if err := vk.Error(res); err != nil {
		log.Printf("WARNING: getting device (%d) extension properties: %s", device, err)
		return false
	}

	requiredExtensions := make(map[string]struct{})
	for _, extensionName := range a.deviceExtensions {
		requiredExtensions[extensionName] = struct{}{}
	}

	for _, extension := range availableExtensions {
		extension.Deref()
		extensionName := vk.ToString(extension.ExtensionName[:])

		delete(requiredExtensions, extensionName+"\x00")
	}

	return len(requiredExtensions) == 0
}

func (a *VulkanTutorialApp) checkValidationSupport() bool {
	var count uint32
	if vk.EnumerateInstanceLayerProperties(&count, nil) != vk.Success {
		return false
	}
	availableLayers := make([]vk.LayerProperties, count)

	if vk.EnumerateInstanceLayerProperties(&count, availableLayers) != vk.Success {
		return false
	}

	availableLayersStr := make([]string, 0, count)
	for _, layer := range availableLayers {
		layer.Deref()

		layerName := vk.ToString(layer.LayerName[:])
		availableLayersStr = append(availableLayersStr, layerName+"\x00")
	}

	for _, validationLayer := range a.validationLayers {
		layerFound := false

		for _, instanceLayer := range availableLayersStr {
			if validationLayer == instanceLayer {
				layerFound = true
				break
			}
		}

		if !layerFound {
			return false
		}
	}

	return true
}

func (a *VulkanTutorialApp) createShaderModule(code []byte) (vk.ShaderModule, error) {
	createInfo := vk.ShaderModuleCreateInfo{
		SType:    vk.StructureTypeShaderModuleCreateInfo,
		CodeSize: uint(len(code)),
		PCode:    unsafer.SliceBytesToUint32(code),
	}

	var shaderModule vk.ShaderModule
	res := vk.CreateShaderModule(a.device, &createInfo, nil, &shaderModule)
	return shaderModule, vk.Error(res)
}

func (a *VulkanTutorialApp) mainLoop() error {
	log.Printf("main loop!\n")

	for !a.window.ShouldClose() {
		err := a.drawFrame()
		if err != nil {
			return fmt.Errorf("error drawing a frame: %w", err)
		}

		glfw.PollEvents()
	}

	vk.DeviceWaitIdle(a.device)

	return nil
}

func (a *VulkanTutorialApp) drawFrame() error {
	fences := []vk.Fence{a.inFlightFences[a.curentFrame]}
	vk.WaitForFences(a.device, 1, fences, vk.True, math.MaxUint64)

	var imageIndex uint32
	res := vk.AcquireNextImage(
		a.device,
		a.swapChain,
		math.MaxUint64,
		a.imageAvailabmeSems[a.curentFrame],
		vk.Fence(vk.NullHandle),
		&imageIndex,
	)
	if res == vk.ErrorOutOfDate {
		a.recreateSwapChain()
		return nil
	} else if res != vk.Success && res != vk.Suboptimal {
		return fmt.Errorf("failed to acquire swap chain image: %w", vk.Error(res))
	}

	// Only reset the fence if we are submitting work.
	vk.ResetFences(a.device, 1, fences)

	commandBuffer := a.commandBuffers[a.curentFrame]

	vk.ResetCommandBuffer(commandBuffer, 0)
	if err := a.recordCommandBuffer(commandBuffer, imageIndex); err != nil {
		return fmt.Errorf("recording command buffer: %w", err)
	}

	a.updateUniformBuffer(a.curentFrame)

	signalSemaphores := []vk.Semaphore{
		a.renderFinishedSems[a.curentFrame],
	}

	submitInfo := vk.SubmitInfo{
		SType:              vk.StructureTypeSubmitInfo,
		WaitSemaphoreCount: 1,
		PWaitSemaphores:    []vk.Semaphore{a.imageAvailabmeSems[a.curentFrame]},
		PWaitDstStageMask: []vk.PipelineStageFlags{
			vk.PipelineStageFlags(vk.PipelineStageColorAttachmentOutputBit),
		},
		CommandBufferCount:   1,
		PCommandBuffers:      []vk.CommandBuffer{commandBuffer},
		PSignalSemaphores:    signalSemaphores,
		SignalSemaphoreCount: uint32(len(signalSemaphores)),
	}

	res = vk.QueueSubmit(
		a.graphicsQueue,
		1,
		[]vk.SubmitInfo{submitInfo},
		a.inFlightFences[a.curentFrame],
	)
	if err := vk.Error(res); err != nil {
		return fmt.Errorf("queue submit error: %w", err)
	}

	swapChains := []vk.Swapchain{
		a.swapChain,
	}

	presentInfo := vk.PresentInfo{
		SType:              vk.StructureTypePresentInfo,
		WaitSemaphoreCount: uint32(len(signalSemaphores)),
		PWaitSemaphores:    signalSemaphores,
		SwapchainCount:     uint32(len(swapChains)),
		PSwapchains:        swapChains,
		PImageIndices:      []uint32{imageIndex},
	}

	res = vk.QueuePresent(a.presentQueue, &presentInfo)
	if res == vk.ErrorOutOfDate || res == vk.Suboptimal || a.frameBufferResized {
		a.frameBufferResized = false
		a.recreateSwapChain()
	} else if res != vk.Success {
		return fmt.Errorf("failed to present swap chain image: %w", vk.Error(res))
	}

	a.curentFrame = (a.curentFrame + 1) % maxFramesInFlight
	return nil
}

func (a *VulkanTutorialApp) updateUniformBuffer(currentImage uint32) {
	frameTime := time.Since(a.startTime)
	ubo := UniformBufferObject{}

	ubo.model.Identity()
	ubo.model.RotateZ(&ubo.model, float32(frameTime.Seconds()))
	ubo.view.LookAt(
		&linmath.Vec3{2, 2, 2},
		&linmath.Vec3{0, 0, 0},
		&linmath.Vec3{0, 0, 1},
	)

	aspectR := float32(a.swapChainExtend.Width) / float32(a.swapChainExtend.Height)
	ubo.proj.Perspective(45, aspectR, 0.1, 10)

	ubo.proj[1][1] *= -1

	vk.Memcopy(a.uniformBuffersMapped[currentImage], unsafer.StructToBytes(&ubo))
}

type Vertex struct {
	pos      linmath.Vec3
	color    linmath.Vec3
	texColor linmath.Vec2
}

func GetVertexSize() uint32 {
	return uint32(unsafe.Sizeof(Vertex{}))
}

func GetVertexBindingDescription() vk.VertexInputBindingDescription {
	bindingDescription := vk.VertexInputBindingDescription{
		Binding:   0,
		Stride:    GetVertexSize(),
		InputRate: vk.VertexInputRateVertex,
	}

	return bindingDescription
}

func GetVertexAttributeDescriptions() [3]vk.VertexInputAttributeDescription {
	attrDescr := [3]vk.VertexInputAttributeDescription{
		{
			Binding:  0,
			Location: 0,
			Format:   vk.FormatR32g32b32Sfloat,
			Offset:   uint32(unsafe.Offsetof(Vertex{}.pos)),
		},
		{
			Binding:  0,
			Location: 1,
			Format:   vk.FormatR32g32b32Sfloat,
			Offset:   uint32(unsafe.Offsetof(Vertex{}.color)),
		},
		{
			Binding:  0,
			Location: 2,
			Format:   vk.FormatR32g32Sfloat,
			Offset:   uint32(unsafe.Offsetof(Vertex{}.texColor)),
		},
	}

	return attrDescr
}

// swapChainSupportDetails describes a present surface. The type is suitable for
// passing around many details of the service between functions.
type swapChainSupportDetails struct {
	capabilities vk.SurfaceCapabilities
	formats      []vk.SurfaceFormat
	presentModes []vk.PresentMode
}

type UniformBufferObject struct {
	model linmath.Mat4x4
	view  linmath.Mat4x4
	proj  linmath.Mat4x4
}

func clamp[T cmp.Ordered](val, min, max T) T {
	if val < min {
		val = min
	}
	if val > max {
		val = max
	}
	return val
}

// QueueFamilyIndices holds the indexes of Vulkan queue families needed by the programs.
type QueueFamilyIndices struct {

	// Graphics is the index of the graphics queue family.
	Graphics optional.Optional[uint32]

	// Present is the index of the queue family used for presenting to the drawing
	// surface.
	Present optional.Optional[uint32]
}

// IsComplete returns true if all families have been set.
func (f *QueueFamilyIndices) IsComplete() bool {
	return f.Graphics.HasValue() && f.Present.HasValue()
}
