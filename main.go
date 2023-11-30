package main

import (
	"cmp"
	"flag"
	"fmt"
	"image"
	"log"
	"math"
	"reflect"
	"runtime"
	"time"
	"unsafe"

	// Used for decoding textures
	"image/draw"
	_ "image/jpeg"

	"vulkan-tutorial/queues"
	"vulkan-tutorial/shaders"
	"vulkan-tutorial/textures"
	"vulkan-tutorial/unsafer"

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
	title             = "Vulkan Tutorial: Hello Triangle"
	maxFramesInFlight = 2
)

func main() {
	flag.Parse()

	app := &HelloTriangleApp{
		width:                  1024,
		height:                 768,
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
				pos:   linmath.Vec2{-0.5, -0.5},
				color: linmath.Vec3{1, 0, 0},
			},
			{
				pos:   linmath.Vec2{0.5, -0.5},
				color: linmath.Vec3{0, 1, 0},
			},
			{
				pos:   linmath.Vec2{0.5, 0.5},
				color: linmath.Vec3{0, 0, 1},
			},
			{
				pos:   linmath.Vec2{-0.5, 0.5},
				color: linmath.Vec3{1, 1, 1},
			},
		},
		indices:            []uint16{0, 1, 2, 2, 3, 0},
		vertexBuffer:       vk.NullBuffer,
		vertexBufferMemory: vk.NullDeviceMemory,
		indexBuffer:        vk.NullBuffer,
		indexBufferMemory:  vk.NullDeviceMemory,
		descriptorPool:     vk.NullDescriptorPool,

		textureImage:       vk.NullImage,
		textureImageMemory: vk.NullDeviceMemory,

		descriptorSetLayout: vk.NullDescriptorSetLayout,
	}
	if err := app.Run(); err != nil {
		log.Fatalf("ERROR: %s", err)
	}
}

// HelloTriangleApp is the first program from vulkan-tutorial.com
type HelloTriangleApp struct {
	width  int
	height int

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
}

// Run runs the vulkan program.
func (h *HelloTriangleApp) Run() error {
	if err := h.initWindow(); err != nil {
		return fmt.Errorf("initWindow: %w", err)
	}
	defer h.cleanWindow()

	if err := h.initVulkan(); err != nil {
		return fmt.Errorf("initVulkan: %w", err)
	}
	defer h.cleanupVulkan()

	if err := h.mainLoop(); err != nil {
		return fmt.Errorf("mainLoop: %w", err)
	}

	if err := h.cleanup(); err != nil {
		return fmt.Errorf("cleanup: %w", err)
	}

	return nil
}

func (h *HelloTriangleApp) initWindow() error {
	if err := glfw.Init(); err != nil {
		return fmt.Errorf("glfw.Init: %w", err)
	}

	glfw.WindowHint(glfw.ClientAPI, glfw.NoAPI)
	// glfw.WindowHint(glfw.Resizable, glfw.False)

	window, err := glfw.CreateWindow(h.width, h.height, title, nil, nil)
	if err != nil {
		return fmt.Errorf("creating window: %w", err)
	}

	window.SetFramebufferSizeCallback(h.frameBufferResizeCallback)

	h.window = window
	return nil
}

func (h *HelloTriangleApp) frameBufferResizeCallback(
	w *glfw.Window,
	width int,
	height int,
) {
	h.frameBufferResized = true
}

func (h *HelloTriangleApp) cleanWindow() {
	h.window.Destroy()
	glfw.Terminate()
}

func (h *HelloTriangleApp) initVulkan() error {
	vk.SetGetInstanceProcAddr(glfw.GetVulkanGetInstanceProcAddress())

	if err := vk.Init(); err != nil {
		return fmt.Errorf("failed to init Vulkan Go: %w", err)
	}

	if err := h.createInsance(); err != nil {
		return fmt.Errorf("createInstance: %w", err)
	}

	if err := h.createSurface(); err != nil {
		return fmt.Errorf("createSurface: %w", err)
	}

	if err := h.pickPhysicalDevice(); err != nil {
		return fmt.Errorf("pickPhysicalDevice: %w", err)
	}

	if err := h.createLogicalDevice(); err != nil {
		return fmt.Errorf("createLogicalDevice: %w", err)
	}

	if err := h.createSwapChain(); err != nil {
		return fmt.Errorf("createSwapChain: %w", err)
	}

	if err := h.createImageViews(); err != nil {
		return fmt.Errorf("createImageViews: %w", err)
	}

	if err := h.createRenderPass(); err != nil {
		return fmt.Errorf("createRenderPass: %w", err)
	}

	if err := h.createDescriptorSetLayout(); err != nil {
		return fmt.Errorf("createDescriptorSetLayout: %w", err)
	}

	if err := h.createGraphicsPipeline(); err != nil {
		return fmt.Errorf("createGraphicsPipeline: %w", err)
	}

	if err := h.createFramebuffers(); err != nil {
		return fmt.Errorf("createFramebuffers: %w", err)
	}

	if err := h.createCommandPool(); err != nil {
		return fmt.Errorf("createCommandPool: %w", err)
	}

	if err := h.createTextureImage(); err != nil {
		return fmt.Errorf("createTextureImage: %w", err)
	}

	if err := h.createVertexBuffer(); err != nil {
		return fmt.Errorf("createVertexBuffer: %w", err)
	}

	if err := h.createIndexBuffer(); err != nil {
		return fmt.Errorf("createIndexBuffer: %w", err)
	}

	if err := h.createUniformBuffers(); err != nil {
		return fmt.Errorf("createUniformBuffers: %w", err)
	}

	if err := h.createDescriptorPool(); err != nil {
		return fmt.Errorf("createDescriptorPool: %w", err)
	}

	if err := h.createDescriptorSets(); err != nil {
		return fmt.Errorf("createDescriptorSets: %w", err)
	}

	if err := h.createCommandBuffer(); err != nil {
		return fmt.Errorf("createCommandBuffer: %w", err)
	}

	if err := h.createSyncObjects(); err != nil {
		return fmt.Errorf("createSyncObjects: %w", err)
	}

	return nil
}

func (h *HelloTriangleApp) cleanupVulkan() {
	for i := 0; i < maxFramesInFlight; i++ {
		vk.DestroySemaphore(h.device, h.imageAvailabmeSems[i], nil)
		vk.DestroySemaphore(h.device, h.renderFinishedSems[i], nil)
		vk.DestroyFence(h.device, h.inFlightFences[i], nil)
	}

	vk.DestroyCommandPool(h.device, h.commandPool, nil)

	vk.DestroyPipeline(h.device, h.graphicsPipline, nil)
	vk.DestroyPipelineLayout(h.device, h.pipelineLayout, nil)

	h.cleanupSwapChain()

	if h.textureImage != vk.NullImage {
		vk.DestroyImage(h.device, h.textureImage, nil)
	}
	if h.textureImageMemory != vk.NullDeviceMemory {
		vk.FreeMemory(h.device, h.textureImageMemory, nil)
	}

	for _, buffer := range h.uniformBuffers {
		vk.DestroyBuffer(h.device, buffer, nil)
	}
	for _, bufferMem := range h.uniformBuffersMemory {
		vk.FreeMemory(h.device, bufferMem, nil)
	}

	if h.descriptorPool != vk.NullDescriptorPool {
		vk.DestroyDescriptorPool(h.device, h.descriptorPool, nil)
	}

	if h.descriptorSetLayout != vk.NullDescriptorSetLayout {
		vk.DestroyDescriptorSetLayout(h.device, h.descriptorSetLayout, nil)
	}

	if h.vertexBuffer != vk.NullBuffer {
		vk.DestroyBuffer(h.device, h.vertexBuffer, nil)
	}
	if h.vertexBufferMemory != vk.NullDeviceMemory {
		vk.FreeMemory(h.device, h.vertexBufferMemory, nil)
	}

	if h.indexBuffer != vk.NullBuffer {
		vk.DestroyBuffer(h.device, h.indexBuffer, nil)
	}
	if h.indexBufferMemory != vk.NullDeviceMemory {
		vk.FreeMemory(h.device, h.indexBufferMemory, nil)
	}

	vk.DestroyRenderPass(h.device, h.renderPass, nil)

	if h.device != vk.Device(vk.NullHandle) {
		vk.DestroyDevice(h.device, nil)
	}
	if h.surface != vk.NullSurface {
		vk.DestroySurface(h.instance, h.surface, nil)
	}
	vk.DestroyInstance(h.instance, nil)
}

func (h *HelloTriangleApp) cleanupSwapChain() {
	for _, frameBuffer := range h.swapChainFramebuffers {
		vk.DestroyFramebuffer(h.device, frameBuffer, nil)
	}

	for _, imageView := range h.swapChainImageViews {
		vk.DestroyImageView(h.device, imageView, nil)
	}

	if h.swapChain != vk.NullSwapchain {
		vk.DestroySwapchain(h.device, h.swapChain, nil)
	}
	h.swapChainImages = nil
	h.swapChainImageViews = nil
}

func (h *HelloTriangleApp) createSurface() error {
	surfacePtr, err := h.window.CreateWindowSurface(h.instance, nil)
	if err != nil {
		return fmt.Errorf("cannot create surface within GLFW window: %w", err)
	}

	h.surface = vk.SurfaceFromPointer(surfacePtr)
	return nil
}

func (h *HelloTriangleApp) pickPhysicalDevice() error {
	var deviceCount uint32
	err := vk.Error(vk.EnumeratePhysicalDevices(h.instance, &deviceCount, nil))
	if err != nil {
		return fmt.Errorf("failed to get the number of physical devices: %w", err)
	}
	if deviceCount == 0 {
		return fmt.Errorf("failed to find GPUs with Vulkan support")
	}

	pDevices := make([]vk.PhysicalDevice, deviceCount)
	err = vk.Error(vk.EnumeratePhysicalDevices(h.instance, &deviceCount, pDevices))
	if err != nil {
		return fmt.Errorf("failed to enumerate the physical devices: %w", err)
	}

	var (
		selectedDevice vk.PhysicalDevice
		score          uint32
	)

	for _, device := range pDevices {
		deviceScore := h.getDeviceScore(device)

		if deviceScore > score {
			selectedDevice = device
			score = deviceScore
		}
	}

	if selectedDevice == vk.PhysicalDevice(vk.NullHandle) {
		return fmt.Errorf("failed to find suitable physical devices")
	}

	h.physicalDevice = selectedDevice
	return nil
}

func (h *HelloTriangleApp) createLogicalDevice() error {
	indices := h.findQueueFamilies(h.physicalDevice)
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

	//!TODO: left for later use
	deviceFeatures := []vk.PhysicalDeviceFeatures{{}}

	createInfo := vk.DeviceCreateInfo{
		SType:            vk.StructureTypeDeviceCreateInfo,
		PEnabledFeatures: deviceFeatures,

		PQueueCreateInfos:    queueCreateInfos,
		QueueCreateInfoCount: uint32(len(queueCreateInfos)),

		EnabledExtensionCount:   uint32(len(h.deviceExtensions)),
		PpEnabledExtensionNames: h.deviceExtensions,
	}

	if h.enableValidationLayers {
		createInfo.PpEnabledLayerNames = h.validationLayers
		createInfo.EnabledLayerCount = uint32(len(h.validationLayers))
	}

	var device vk.Device
	err := vk.Error(vk.CreateDevice(h.physicalDevice, &createInfo, nil, &device))
	if err != nil {
		return fmt.Errorf("failed to create logical device: %w", err)
	}
	h.device = device

	var graphicsQueue vk.Queue
	vk.GetDeviceQueue(h.device, indices.Graphics.Get(), 0, &graphicsQueue)
	h.graphicsQueue = graphicsQueue

	var presentQueue vk.Queue
	vk.GetDeviceQueue(h.device, indices.Present.Get(), 0, &presentQueue)
	h.presentQueue = presentQueue

	return nil
}

func (h *HelloTriangleApp) createSwapChain() error {
	swapChainSupport := h.querySwapChainSupport(h.physicalDevice)

	surfaceFormat := h.chooseSwapSurfaceFormat(swapChainSupport.formats)
	presentMode := h.chooseSwapPresentMode(swapChainSupport.presentModes)
	extend := h.chooseSwapExtend(swapChainSupport.capabilities)

	imageCount := swapChainSupport.capabilities.MinImageCount + 1
	if swapChainSupport.capabilities.MaxImageCount > 0 &&
		imageCount > swapChainSupport.capabilities.MaxImageCount {
		imageCount = swapChainSupport.capabilities.MaxImageCount
	}

	createInfo := vk.SwapchainCreateInfo{
		SType:            vk.StructureTypeSwapchainCreateInfo,
		Surface:          h.surface,
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

	indices := h.findQueueFamilies(h.physicalDevice)
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
	res := vk.CreateSwapchain(h.device, &createInfo, nil, &swapChain)
	if err := vk.Error(res); err != nil {
		return fmt.Errorf("failed to create swap chain: %w", err)
	}
	h.swapChain = swapChain

	var imagesCount uint32
	vk.GetSwapchainImages(h.device, h.swapChain, &imagesCount, nil)

	images := make([]vk.Image, imagesCount)
	vk.GetSwapchainImages(h.device, h.swapChain, &imagesCount, images)

	h.swapChainImages = images

	h.swapChainImageFormat = surfaceFormat.Format
	h.swapChainExtend = extend

	return nil
}

func (h *HelloTriangleApp) recreateSwapChain() error {
	for true {
		width, height := h.window.GetFramebufferSize()
		if width != 0 || height != 0 {
			break
		}

		glfw.WaitEvents()
	}

	vk.DeviceWaitIdle(h.device)

	h.cleanupSwapChain()

	if err := h.createSwapChain(); err != nil {
		return fmt.Errorf("createSwapChain: %w", err)
	}
	if err := h.createImageViews(); err != nil {
		return fmt.Errorf("createImageViews: %w", err)
	}
	if err := h.createFramebuffers(); err != nil {
		return fmt.Errorf("createFramebuffers: %w", err)
	}

	return nil
}

func (h *HelloTriangleApp) createImageViews() error {
	for i, swapChainImage := range h.swapChainImages {
		swapChainImage := swapChainImage
		createInfo := vk.ImageViewCreateInfo{
			SType:    vk.StructureTypeImageViewCreateInfo,
			Image:    swapChainImage,
			ViewType: vk.ImageViewType2d,
			Format:   h.swapChainImageFormat,
			Components: vk.ComponentMapping{
				R: vk.ComponentSwizzleIdentity,
				G: vk.ComponentSwizzleIdentity,
				B: vk.ComponentSwizzleIdentity,
				A: vk.ComponentSwizzleIdentity,
			},
			SubresourceRange: vk.ImageSubresourceRange{
				AspectMask:     vk.ImageAspectFlags(vk.ImageAspectColorBit),
				BaseMipLevel:   0,
				LevelCount:     1,
				BaseArrayLayer: 0,
				LayerCount:     1,
			},
		}

		var imageView vk.ImageView
		res := vk.CreateImageView(h.device, &createInfo, nil, &imageView)
		if err := vk.Error(res); err != nil {
			return fmt.Errorf("failed to create image %d: %w", i, err)
		}

		h.swapChainImageViews = append(h.swapChainImageViews, imageView)
	}

	return nil
}

func (h *HelloTriangleApp) createRenderPass() error {
	colorAttachment := vk.AttachmentDescription{
		Format:         h.swapChainImageFormat,
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
		PipelineBindPoint:    vk.PipelineBindPointGraphics,
		ColorAttachmentCount: 1,
		PColorAttachments:    []vk.AttachmentReference{colorAttachmentRef},
	}

	dependency := vk.SubpassDependency{
		SrcSubpass:    vk.SubpassExternal,
		DstSubpass:    0,
		SrcStageMask:  vk.PipelineStageFlags(vk.PipelineStageColorAttachmentOutputBit),
		SrcAccessMask: 0,
		DstStageMask:  vk.PipelineStageFlags(vk.PipelineStageColorAttachmentOutputBit),
		DstAccessMask: vk.AccessFlags(vk.AccessColorAttachmentWriteBit),
	}

	rederPassInfo := vk.RenderPassCreateInfo{
		SType:           vk.StructureTypeRenderPassCreateInfo,
		AttachmentCount: 1,
		PAttachments:    []vk.AttachmentDescription{colorAttachment},
		SubpassCount:    1,
		PSubpasses:      []vk.SubpassDescription{subpass},
		DependencyCount: 1,
		PDependencies:   []vk.SubpassDependency{dependency},
	}

	var renderPass vk.RenderPass
	res := vk.CreateRenderPass(h.device, &rederPassInfo, nil, &renderPass)
	if err := vk.Error(res); err != nil {
		return fmt.Errorf("failed to create render pass: %w", err)
	}
	h.renderPass = renderPass

	return nil
}

func (h *HelloTriangleApp) createDescriptorSetLayout() error {
	uboLayoutBinding := vk.DescriptorSetLayoutBinding{
		Binding:            0,
		DescriptorType:     vk.DescriptorTypeUniformBuffer,
		DescriptorCount:    1,
		StageFlags:         vk.ShaderStageFlags(vk.ShaderStageVertexBit),
		PImmutableSamplers: nil,
	}

	layoutInfo := vk.DescriptorSetLayoutCreateInfo{
		SType:        vk.StructureTypeDescriptorSetLayoutCreateInfo,
		BindingCount: 1,
		PBindings:    []vk.DescriptorSetLayoutBinding{uboLayoutBinding},
	}

	var descriptorSetLayout vk.DescriptorSetLayout
	res := vk.CreateDescriptorSetLayout(h.device, &layoutInfo, nil, &descriptorSetLayout)
	if res != vk.Success {
		return fmt.Errorf("creating descriptor set layout: %w", vk.Error(res))
	}
	h.descriptorSetLayout = descriptorSetLayout

	return nil
}

func (h *HelloTriangleApp) createGraphicsPipeline() error {
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

	vertexShaderModule, err := h.createShaderModule(vertShaderCode)
	if err != nil {
		return fmt.Errorf("creating vertex shader module: %w", err)
	}
	defer vk.DestroyShaderModule(h.device, vertexShaderModule, nil)

	fragmentShaderModule, err := h.createShaderModule(fragShaderCode)
	if err != nil {
		return fmt.Errorf("creating fragment shader module: %w", err)
	}
	defer vk.DestroyShaderModule(h.device, fragmentShaderModule, nil)

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
		Width:    float32(h.swapChainExtend.Width),
		Height:   float32(h.swapChainExtend.Height),
		MinDepth: 0,
		MaxDepth: 1,
	}

	scissor := vk.Rect2D{
		Offset: vk.Offset2D{X: 0, Y: 0},
		Extent: h.swapChainExtend,
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
		PSetLayouts:    []vk.DescriptorSetLayout{h.descriptorSetLayout},
	}

	var pipelineLayout vk.PipelineLayout
	res := vk.CreatePipelineLayout(h.device, &pipelineLayoutInfo, nil, &pipelineLayout)
	if err := vk.Error(res); err != nil {
		return fmt.Errorf("failed to create pipeline layout: %w", err)
	}
	h.pipelineLayout = pipelineLayout

	pipelineInfo := vk.GraphicsPipelineCreateInfo{
		SType:               vk.StructureTypeGraphicsPipelineCreateInfo,
		StageCount:          uint32(len(shaderStages)),
		PStages:             shaderStages,
		PVertexInputState:   &vertexInputInfo,
		PInputAssemblyState: &inputAssembly,
		PViewportState:      &viewportState,
		PRasterizationState: &rasterizer,
		PMultisampleState:   &multisampling,
		PDepthStencilState:  nil,
		PColorBlendState:    &colorBlending,
		PDynamicState:       &dynamicState,
		Layout:              h.pipelineLayout,
		RenderPass:          h.renderPass,
		Subpass:             0,
		BasePipelineHandle:  vk.Pipeline(vk.NullHandle),
		BasePipelineIndex:   -1,
	}

	pipelines := make([]vk.Pipeline, 1)
	res = vk.CreateGraphicsPipelines(
		h.device,
		vk.PipelineCache(vk.NullHandle),
		1,
		[]vk.GraphicsPipelineCreateInfo{pipelineInfo},
		nil,
		pipelines,
	)
	if err := vk.Error(res); err != nil {
		return fmt.Errorf("failed to create graphics pipeline: %w", err)
	}
	h.graphicsPipline = pipelines[0]

	return nil
}

func (h *HelloTriangleApp) createFramebuffers() error {
	h.swapChainFramebuffers = make([]vk.Framebuffer, len(h.swapChainImageViews))

	for i, swapChainView := range h.swapChainImageViews {
		swapChainView := swapChainView

		attachments := []vk.ImageView{
			swapChainView,
		}

		frameBufferInfo := vk.FramebufferCreateInfo{
			SType:           vk.StructureTypeFramebufferCreateInfo,
			RenderPass:      h.renderPass,
			AttachmentCount: 1,
			PAttachments:    attachments,
			Width:           h.swapChainExtend.Width,
			Height:          h.swapChainExtend.Height,
			Layers:          1,
		}

		var frameBuffer vk.Framebuffer
		res := vk.CreateFramebuffer(h.device, &frameBufferInfo, nil, &frameBuffer)
		if err := vk.Error(res); err != nil {
			return fmt.Errorf("failed to create frame buffer %d: %w", i, err)
		}

		h.swapChainFramebuffers[i] = frameBuffer
	}

	return nil
}

func (h *HelloTriangleApp) createCommandPool() error {
	queueFamilyIndices := h.findQueueFamilies(h.physicalDevice)
	poolInfo := vk.CommandPoolCreateInfo{
		SType: vk.StructureTypeCommandPoolCreateInfo,
		Flags: vk.CommandPoolCreateFlags(
			vk.CommandPoolCreateResetCommandBufferBit,
		),
		QueueFamilyIndex: queueFamilyIndices.Graphics.Get(),
	}

	var commandPool vk.CommandPool
	res := vk.CreateCommandPool(h.device, &poolInfo, nil, &commandPool)
	if err := vk.Error(res); err != nil {
		return fmt.Errorf("failed to create command pool: %w", err)
	}
	h.commandPool = commandPool

	return nil
}

func (h *HelloTriangleApp) createVertexBuffer() error {
	bufferSize := vk.DeviceSize(uint32(len(h.vertices)) * GetVertexSize())

	// Create the staging buffer
	var (
		stagingBuffer       vk.Buffer
		stagingBufferMemory vk.DeviceMemory
	)
	err := h.createBuffer(
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
		vk.DestroyBuffer(h.device, stagingBuffer, nil)
		vk.FreeMemory(h.device, stagingBufferMemory, nil)
	}()

	// Copy the data from host to staging buffer
	var pData unsafe.Pointer
	vk.MapMemory(h.device, stagingBufferMemory, 0, bufferSize, 0, &pData)

	bytesSlice := unsafer.SliceToBytes(h.vertices)

	vk.Memcopy(pData, bytesSlice)
	vk.UnmapMemory(h.device, stagingBufferMemory)

	// Create the device local buffer
	var (
		vertexBuffer       vk.Buffer
		vertexBufferMemory vk.DeviceMemory
	)

	err = h.createBuffer(
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
	h.vertexBuffer = vertexBuffer
	h.vertexBufferMemory = vertexBufferMemory

	// Copy data from the staging buffer to the device local buffer which is our
	// vertex buffer
	if err := h.copyBuffer(stagingBuffer, h.vertexBuffer, bufferSize); err != nil {
		return fmt.Errorf("failed to copy staging buffer into vertex: %w", err)
	}

	return nil
}

func (h *HelloTriangleApp) createIndexBuffer() error {
	bufferSize := vk.DeviceSize(uint32(len(h.indices)) * uint32(unsafe.Sizeof(h.indices[0])))

	// Create the staging buffer
	var (
		stagingBuffer       vk.Buffer
		stagingBufferMemory vk.DeviceMemory
	)
	err := h.createBuffer(
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
		vk.DestroyBuffer(h.device, stagingBuffer, nil)
		vk.FreeMemory(h.device, stagingBufferMemory, nil)
	}()

	// Copy the data from host to staging buffer
	var pData unsafe.Pointer
	vk.MapMemory(h.device, stagingBufferMemory, 0, bufferSize, 0, &pData)

	bytesSlice := unsafer.SliceToBytes(h.indices)

	vk.Memcopy(pData, bytesSlice)
	vk.UnmapMemory(h.device, stagingBufferMemory)

	// Create the device local buffer
	var (
		indexBuffer       vk.Buffer
		indexBufferMemory vk.DeviceMemory
	)

	err = h.createBuffer(
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
	h.indexBuffer = indexBuffer
	h.indexBufferMemory = indexBufferMemory

	// Copy data from the staging buffer to the device local buffer which is our
	// index buffer
	if err := h.copyBuffer(stagingBuffer, h.indexBuffer, bufferSize); err != nil {
		return fmt.Errorf("failed to copy staging buffer into index: %w", err)
	}

	return nil
}

func (h *HelloTriangleApp) copyBuffer(
	srcBuffer vk.Buffer,
	dstBuffer vk.Buffer,
	size vk.DeviceSize,
) error {
	commandBuffer, err := h.beginSingleTimeCommands()
	if err != nil {
		return fmt.Errorf("failed to begin single time commands: %w", err)
	}

	copyRegion := vk.BufferCopy{
		SrcOffset: 0,
		DstOffset: 0,
		Size:      size,
	}

	vk.CmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, []vk.BufferCopy{copyRegion})

	return h.endSingleTimeCommands(commandBuffer)
}

func (h *HelloTriangleApp) beginSingleTimeCommands() (vk.CommandBuffer, error) {
	allocInfo := vk.CommandBufferAllocateInfo{
		SType:              vk.StructureTypeCommandBufferAllocateInfo,
		Level:              vk.CommandBufferLevelPrimary,
		CommandPool:        h.commandPool,
		CommandBufferCount: 1,
	}

	commandBuffers := make([]vk.CommandBuffer, 1)
	res := vk.AllocateCommandBuffers(
		h.device,
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

func (h *HelloTriangleApp) endSingleTimeCommands(commandBuffer vk.CommandBuffer) error {
	commandBuffers := []vk.CommandBuffer{commandBuffer}

	defer func() {
		vk.FreeCommandBuffers(h.device, h.commandPool, 1, commandBuffers)
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

	res = vk.QueueSubmit(h.graphicsQueue, 1, []vk.SubmitInfo{submitInfo}, vk.NullFence)
	if res != vk.Success {
		return fmt.Errorf("failed to submit to graphics queue: %w", vk.Error(res))
	}

	res = vk.QueueWaitIdle(h.graphicsQueue)
	if res != vk.Success {
		return fmt.Errorf("failed to wait on graphics queue idle: %w", vk.Error(res))
	}

	return nil
}

func (h *HelloTriangleApp) transitionImageLayout(
	image vk.Image,
	format vk.Format,
	oldLayout vk.ImageLayout,
	newLayout vk.ImageLayout,
) error {
	commandBuffer, err := h.beginSingleTimeCommands()
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

	return h.endSingleTimeCommands(commandBuffer)
}

func (h *HelloTriangleApp) copyBufferToImage(
	buffer vk.Buffer,
	image vk.Image,
	width, height uint32,
) error {
	commandBuffer, err := h.beginSingleTimeCommands()
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

	return h.endSingleTimeCommands(commandBuffer)
}

func (h *HelloTriangleApp) createUniformBuffers() error {
	bufferSize := vk.DeviceSize(unsafe.Sizeof(UniformBufferObject{}))

	for i := 0; i < maxFramesInFlight; i++ {
		var (
			buffer       vk.Buffer
			bufferMemory vk.DeviceMemory
		)
		err := h.createBuffer(
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

		h.uniformBuffers = append(h.uniformBuffers, buffer)
		h.uniformBuffersMemory = append(h.uniformBuffersMemory, bufferMemory)

		var pData unsafe.Pointer
		vk.MapMemory(h.device, h.uniformBuffersMemory[i], 0, bufferSize, 0, &pData)
		h.uniformBuffersMapped = append(h.uniformBuffersMapped, pData)
	}

	return nil
}

func (h *HelloTriangleApp) createDescriptorPool() error {
	poolSize := vk.DescriptorPoolSize{
		Type:            vk.DescriptorTypeUniformBuffer,
		DescriptorCount: maxFramesInFlight,
	}

	poolInfo := vk.DescriptorPoolCreateInfo{
		SType:         vk.StructureTypeDescriptorPoolCreateInfo,
		PoolSizeCount: 1,
		PPoolSizes:    []vk.DescriptorPoolSize{poolSize},
		MaxSets:       maxFramesInFlight,
	}

	var descriptorPool vk.DescriptorPool
	res := vk.CreateDescriptorPool(h.device, &poolInfo, nil, &descriptorPool)
	if res != vk.Success {
		return fmt.Errorf("failed to create descriptor pool: %w", vk.Error(res))
	}
	h.descriptorPool = descriptorPool

	return nil
}

func (h *HelloTriangleApp) createDescriptorSets() error {
	layouts := []vk.DescriptorSetLayout{
		h.descriptorSetLayout,
		h.descriptorSetLayout,
	}

	allocInfo := vk.DescriptorSetAllocateInfo{
		SType:              vk.StructureTypeDescriptorSetAllocateInfo,
		DescriptorPool:     h.descriptorPool,
		DescriptorSetCount: maxFramesInFlight,
		PSetLayouts:        layouts,
	}

	h.descriptorSets = make([]vk.DescriptorSet, maxFramesInFlight)

	res := vk.AllocateDescriptorSets(h.device, &allocInfo, &h.descriptorSets[0])
	if res != vk.Success {
		return fmt.Errorf("failed to allocate descriptor set: %w", vk.Error(res))
	}

	for i := 0; i < maxFramesInFlight; i++ {
		bufferInfo := vk.DescriptorBufferInfo{
			Buffer: h.uniformBuffers[i],
			Offset: 0,
			Range:  vk.DeviceSize(vk.WholeSize),
		}

		descriptorWrite := vk.WriteDescriptorSet{
			SType:           vk.StructureTypeWriteDescriptorSet,
			DstSet:          h.descriptorSets[i],
			DstBinding:      0,
			DstArrayElement: 0,
			DescriptorType:  vk.DescriptorTypeUniformBuffer,
			DescriptorCount: 1,
			PBufferInfo:     []vk.DescriptorBufferInfo{bufferInfo},
		}

		writes := []vk.WriteDescriptorSet{descriptorWrite}
		vk.UpdateDescriptorSets(h.device, 1, writes, 0, nil)
	}

	return nil
}

func (h *HelloTriangleApp) createBuffer(
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

	res := vk.CreateBuffer(h.device, &bufferInfo, nil, buffer)
	if res != vk.Success {
		return fmt.Errorf("failed to create vertex buffer: %w", vk.Error(res))
	}

	var memRequirements vk.MemoryRequirements
	vk.GetBufferMemoryRequirements(h.device, *buffer, &memRequirements)
	memRequirements.Deref()

	memTypeIndex, err := h.findMemoryType(memRequirements.MemoryTypeBits, properties)
	if err != nil {
		return err
	}

	allocInfo := vk.MemoryAllocateInfo{
		SType:           vk.StructureTypeMemoryAllocateInfo,
		AllocationSize:  memRequirements.Size,
		MemoryTypeIndex: memTypeIndex,
	}

	res = vk.AllocateMemory(h.device, &allocInfo, nil, bufferMemory)
	if res != vk.Success {
		return fmt.Errorf("failed to allocate vertex buffer memory: %s", vk.Error(res))
	}

	res = vk.BindBufferMemory(h.device, *buffer, *bufferMemory, 0)
	if res != vk.Success {
		return fmt.Errorf("failed to bind buffer memory: %w", vk.Error(res))
	}

	return nil
}

func (h *HelloTriangleApp) createImage(
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

	res := vk.CreateImage(h.device, &imageInfo, nil, image)
	if res != vk.Success {
		return fmt.Errorf("failed to create an image: %w", vk.Error(res))
	}

	var memRequirements vk.MemoryRequirements
	vk.GetImageMemoryRequirements(h.device, *image, &memRequirements)
	memRequirements.Deref()

	memTypeIndex, err := h.findMemoryType(memRequirements.MemoryTypeBits, properties)
	if err != nil {
		return err
	}

	allocInfo := vk.MemoryAllocateInfo{
		SType:           vk.StructureTypeMemoryAllocateInfo,
		AllocationSize:  memRequirements.Size,
		MemoryTypeIndex: memTypeIndex,
	}

	res = vk.AllocateMemory(h.device, &allocInfo, nil, imageMemory)
	if res != vk.Success {
		return fmt.Errorf("failed to allocate image buffer memory: %s", vk.Error(res))
	}

	res = vk.BindImageMemory(h.device, *image, *imageMemory, 0)
	if res != vk.Success {
		return fmt.Errorf("failed to bind image memory: %w", vk.Error(res))
	}

	return nil
}

func (h *HelloTriangleApp) findMemoryType(
	typeFilter uint32,
	properties vk.MemoryPropertyFlags,
) (uint32, error) {
	var memProperties vk.PhysicalDeviceMemoryProperties
	vk.GetPhysicalDeviceMemoryProperties(h.physicalDevice, &memProperties)
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

func (h *HelloTriangleApp) createTextureImage() error {
	fh, err := textures.FS.Open("texture.jpg")
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

	err = h.createBuffer(
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
		vk.DestroyBuffer(h.device, staginbBuffer, nil)
		vk.FreeMemory(h.device, stagingBufferMemory, nil)
	}()

	var pData unsafe.Pointer
	vk.MapMemory(h.device, stagingBufferMemory, 0, imgSize, 0, &pData)
	defer vk.UnmapMemory(h.device, stagingBufferMemory)

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

	err = h.createImage(
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
	h.textureImage = textureImage
	h.textureImageMemory = textureImageMemory

	err = h.transitionImageLayout(
		h.textureImage,
		vk.FormatR8g8b8a8Srgb,
		vk.ImageLayoutUndefined,
		vk.ImageLayoutTransferDstOptimal,
	)
	if err != nil {
		return fmt.Errorf("transition image layout: %w", err)
	}

	err = h.copyBufferToImage(staginbBuffer, h.textureImage, texWidth, texHeight)
	if err != nil {
		return fmt.Errorf("copying buffer to image: %w", err)
	}

	err = h.transitionImageLayout(
		h.textureImage,
		vk.FormatR8g8b8a8Srgb,
		vk.ImageLayoutTransferDstOptimal,
		vk.ImageLayoutShaderReadOnlyOptimal,
	)
	if err != nil {
		return fmt.Errorf("transitioning to read only optimal layout: %w", err)
	}

	return nil
}

func (h *HelloTriangleApp) createCommandBuffer() error {
	allocInfo := vk.CommandBufferAllocateInfo{
		SType:              vk.StructureTypeCommandBufferAllocateInfo,
		CommandPool:        h.commandPool,
		Level:              vk.CommandBufferLevelPrimary,
		CommandBufferCount: 2,
	}

	commandBuffers := make([]vk.CommandBuffer, 2)
	res := vk.AllocateCommandBuffers(h.device, &allocInfo, commandBuffers)
	if err := vk.Error(res); err != nil {
		return fmt.Errorf("failed to allocate command buffer: %w", err)
	}
	h.commandBuffers = commandBuffers

	return nil
}

func (h *HelloTriangleApp) recordCommandBuffer(
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

	clearColor := vk.NewClearValue([]float32{0, 0, 0, 1})

	renderPassInfo := vk.RenderPassBeginInfo{
		SType:       vk.StructureTypeRenderPassBeginInfo,
		RenderPass:  h.renderPass,
		Framebuffer: h.swapChainFramebuffers[imageIndex],
		RenderArea: vk.Rect2D{
			Offset: vk.Offset2D{
				X: 0,
				Y: 0,
			},
			Extent: h.swapChainExtend,
		},
		ClearValueCount: 1,
		PClearValues:    []vk.ClearValue{clearColor},
	}

	vk.CmdBeginRenderPass(commandBuffer, &renderPassInfo, vk.SubpassContentsInline)
	vk.CmdBindPipeline(commandBuffer, vk.PipelineBindPointGraphics, h.graphicsPipline)

	vertexBuffers := []vk.Buffer{h.vertexBuffer}
	offsets := []vk.DeviceSize{0}
	vk.CmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets)

	vk.CmdBindIndexBuffer(commandBuffer, h.indexBuffer, 0, vk.IndexTypeUint16)

	viewport := vk.Viewport{
		X: 0, Y: 0,
		Width:    float32(h.swapChainExtend.Width),
		Height:   float32(h.swapChainExtend.Height),
		MinDepth: 0,
		MaxDepth: 1,
	}
	vk.CmdSetViewport(commandBuffer, 0, 1, []vk.Viewport{viewport})

	scissor := vk.Rect2D{
		Offset: vk.Offset2D{X: 0, Y: 0},
		Extent: h.swapChainExtend,
	}
	vk.CmdSetScissor(commandBuffer, 0, 1, []vk.Rect2D{scissor})

	vk.CmdBindDescriptorSets(
		commandBuffer,
		vk.PipelineBindPointGraphics,
		h.pipelineLayout,
		0,
		1,
		[]vk.DescriptorSet{h.descriptorSets[h.curentFrame]},
		0,
		nil,
	)

	vk.CmdDrawIndexed(commandBuffer, uint32(len(h.indices)), 1, 0, 0, 0)
	vk.CmdEndRenderPass(commandBuffer)

	if err := vk.Error(vk.EndCommandBuffer(commandBuffer)); err != nil {
		return fmt.Errorf("recording commands to buffer failed: %w", err)
	}
	return nil
}

func (h *HelloTriangleApp) createSyncObjects() error {
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
			vk.CreateSemaphore(h.device, &semaphoreInfo, nil, &imageAvailabmeSem),
		); err != nil {
			return fmt.Errorf("failed to create imageAvailabmeSem: %w", err)
		}
		h.imageAvailabmeSems = append(h.imageAvailabmeSems, imageAvailabmeSem)

		var renderFinishedSem vk.Semaphore
		if err := vk.Error(
			vk.CreateSemaphore(h.device, &semaphoreInfo, nil, &renderFinishedSem),
		); err != nil {
			return fmt.Errorf("failed to create renderFinishedSem: %w", err)
		}
		h.renderFinishedSems = append(h.renderFinishedSems, renderFinishedSem)

		var fence vk.Fence
		if err := vk.Error(
			vk.CreateFence(h.device, &fenceInfo, nil, &fence),
		); err != nil {
			return fmt.Errorf("failed to create inFlightFence: %w", err)
		}
		h.inFlightFences = append(h.inFlightFences, fence)
	}

	return nil
}

func (h *HelloTriangleApp) createInsance() error {
	if h.enableValidationLayers && !h.checkValidationSupport() {
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

	if h.enableValidationLayers {
		createInfo.EnabledLayerCount = uint32(len(h.validationLayers))
		createInfo.PpEnabledLayerNames = h.validationLayers
	}

	var instance vk.Instance
	if err := vk.Error(vk.CreateInstance(&createInfo, nil, &instance)); err != nil {
		return fmt.Errorf("failed to create Vulkan instance: %w", err)
	}

	h.instance = instance
	return nil
}

// findQueueFamilies returns a FamilyIndeces populated with Vulkan queue families needed
// by the program.
func (h *HelloTriangleApp) findQueueFamilies(device vk.PhysicalDevice) queues.FamilyIndices {
	indices := queues.FamilyIndices{}

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
			vk.GetPhysicalDeviceSurfaceSupport(device, uint32(i), h.surface, &hasPresent),
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

func (h *HelloTriangleApp) querySwapChainSupport(
	device vk.PhysicalDevice,
) swapChainSupportDetails {
	details := swapChainSupportDetails{}

	var capabilities vk.SurfaceCapabilities
	res := vk.GetPhysicalDeviceSurfaceCapabilities(device, h.surface, &capabilities)
	if err := vk.Error(res); err != nil {
		panic(fmt.Sprintf("failed to query device surface capabilities: %s", err))
	}
	capabilities.Deref()
	capabilities.CurrentExtent.Deref()
	capabilities.MinImageExtent.Deref()
	capabilities.MaxImageExtent.Deref()

	details.capabilities = capabilities

	var formatCount uint32
	res = vk.GetPhysicalDeviceSurfaceFormats(device, h.surface, &formatCount, nil)
	if err := vk.Error(res); err != nil {
		panic(fmt.Sprintf("failed to query device surface formats: %s", err))
	}

	if formatCount != 0 {
		formats := make([]vk.SurfaceFormat, formatCount)
		vk.GetPhysicalDeviceSurfaceFormats(device, h.surface, &formatCount, formats)
		for _, format := range formats {
			format.Deref()
			details.formats = append(details.formats, format)
		}
	}

	var presentModeCount uint32
	res = vk.GetPhysicalDeviceSurfacePresentModes(
		device, h.surface, &presentModeCount, nil,
	)
	if err := vk.Error(res); err != nil {
		panic(fmt.Sprintf("failed to query device surface present modes: %s", err))
	}

	if presentModeCount != 0 {
		presentModes := make([]vk.PresentMode, presentModeCount)
		vk.GetPhysicalDeviceSurfacePresentModes(
			device, h.surface, &presentModeCount, presentModes,
		)
		details.presentModes = presentModes
	}

	return details
}

// getDeviceScore returns how suitable is this device for the current program.
// Bigger score means better. Zero or negative means the device cannot be used.
func (h *HelloTriangleApp) getDeviceScore(device vk.PhysicalDevice) uint32 {
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

	if !h.isDeviceSuitable(device) {
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

func (h *HelloTriangleApp) isDeviceSuitable(device vk.PhysicalDevice) bool {
	indices := h.findQueueFamilies(device)
	extensionsSupported := h.checkDeviceExtensionSupport(device)

	swapChainAdequate := false
	if extensionsSupported {
		swapChainSupport := h.querySwapChainSupport(device)
		swapChainAdequate = len(swapChainSupport.formats) > 0 &&
			len(swapChainSupport.presentModes) > 0
	}

	return indices.IsComplete() && extensionsSupported && swapChainAdequate
}

func (h *HelloTriangleApp) chooseSwapSurfaceFormat(
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

func (h *HelloTriangleApp) chooseSwapPresentMode(
	available []vk.PresentMode,
) vk.PresentMode {
	for _, mode := range available {
		if mode == vk.PresentModeMailbox {
			return mode
		}
	}

	return vk.PresentModeFifo
}

func (h *HelloTriangleApp) chooseSwapExtend(
	capabilities vk.SurfaceCapabilities,
) vk.Extent2D {
	if capabilities.CurrentExtent.Width != math.MaxUint32 {
		return capabilities.CurrentExtent
	}

	width, height := h.window.GetFramebufferSize()

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

func (h *HelloTriangleApp) checkDeviceExtensionSupport(device vk.PhysicalDevice) bool {
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
	for _, extensionName := range h.deviceExtensions {
		requiredExtensions[extensionName] = struct{}{}
	}

	for _, extension := range availableExtensions {
		extension.Deref()
		extensionName := vk.ToString(extension.ExtensionName[:])

		delete(requiredExtensions, extensionName+"\x00")
	}

	return len(requiredExtensions) == 0
}

func (h *HelloTriangleApp) checkValidationSupport() bool {
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

	for _, validationLayer := range h.validationLayers {
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

func (h *HelloTriangleApp) createShaderModule(code []byte) (vk.ShaderModule, error) {
	createInfo := vk.ShaderModuleCreateInfo{
		SType:    vk.StructureTypeShaderModuleCreateInfo,
		CodeSize: uint(len(code)),
		PCode:    repackUint32(code),
	}

	var shaderModule vk.ShaderModule
	res := vk.CreateShaderModule(h.device, &createInfo, nil, &shaderModule)
	return shaderModule, vk.Error(res)
}

func (h *HelloTriangleApp) mainLoop() error {
	log.Printf("main loop!\n")

	for !h.window.ShouldClose() {
		err := h.drawFrame()
		if err != nil {
			return fmt.Errorf("error drawing a frame: %w", err)
		}

		glfw.PollEvents()
	}

	vk.DeviceWaitIdle(h.device)

	return nil
}

func (h *HelloTriangleApp) drawFrame() error {
	fences := []vk.Fence{h.inFlightFences[h.curentFrame]}
	vk.WaitForFences(h.device, 1, fences, vk.True, math.MaxUint64)

	var imageIndex uint32
	res := vk.AcquireNextImage(
		h.device,
		h.swapChain,
		math.MaxUint64,
		h.imageAvailabmeSems[h.curentFrame],
		vk.Fence(vk.NullHandle),
		&imageIndex,
	)
	if res == vk.ErrorOutOfDate {
		h.recreateSwapChain()
		return nil
	} else if res != vk.Success && res != vk.Suboptimal {
		return fmt.Errorf("failed to acquire swap chain image: %w", vk.Error(res))
	}

	// Only reset the fence if we are submitting work.
	vk.ResetFences(h.device, 1, fences)

	commandBuffer := h.commandBuffers[h.curentFrame]

	vk.ResetCommandBuffer(commandBuffer, 0)
	if err := h.recordCommandBuffer(commandBuffer, imageIndex); err != nil {
		return fmt.Errorf("recording command buffer: %w", err)
	}

	h.updateUniformBuffer(h.curentFrame)

	signalSemaphores := []vk.Semaphore{
		h.renderFinishedSems[h.curentFrame],
	}

	submitInfo := vk.SubmitInfo{
		SType:              vk.StructureTypeSubmitInfo,
		WaitSemaphoreCount: 1,
		PWaitSemaphores:    []vk.Semaphore{h.imageAvailabmeSems[h.curentFrame]},
		PWaitDstStageMask: []vk.PipelineStageFlags{
			vk.PipelineStageFlags(vk.PipelineStageColorAttachmentOutputBit),
		},
		CommandBufferCount:   1,
		PCommandBuffers:      []vk.CommandBuffer{commandBuffer},
		PSignalSemaphores:    signalSemaphores,
		SignalSemaphoreCount: uint32(len(signalSemaphores)),
	}

	res = vk.QueueSubmit(
		h.graphicsQueue,
		1,
		[]vk.SubmitInfo{submitInfo},
		h.inFlightFences[h.curentFrame],
	)
	if err := vk.Error(res); err != nil {
		return fmt.Errorf("queue submit error: %w", err)
	}

	swapChains := []vk.Swapchain{
		h.swapChain,
	}

	presentInfo := vk.PresentInfo{
		SType:              vk.StructureTypePresentInfo,
		WaitSemaphoreCount: uint32(len(signalSemaphores)),
		PWaitSemaphores:    signalSemaphores,
		SwapchainCount:     uint32(len(swapChains)),
		PSwapchains:        swapChains,
		PImageIndices:      []uint32{imageIndex},
	}

	res = vk.QueuePresent(h.presentQueue, &presentInfo)
	if res == vk.ErrorOutOfDate || res == vk.Suboptimal || h.frameBufferResized {
		h.frameBufferResized = false
		h.recreateSwapChain()
	} else if res != vk.Success {
		return fmt.Errorf("failed to present swap chain image: %w", vk.Error(res))
	}

	h.curentFrame = (h.curentFrame + 1) % maxFramesInFlight
	return nil
}

func (h *HelloTriangleApp) updateUniformBuffer(currentImage uint32) {
	frameTime := time.Since(h.startTime)
	ubo := UniformBufferObject{}

	ubo.model.Identity()
	ubo.model.RotateZ(&ubo.model, float32(frameTime.Seconds()))
	ubo.view.LookAt(
		&linmath.Vec3{2, 2, 2},
		&linmath.Vec3{0, 0, 0},
		&linmath.Vec3{0, 0, 1},
	)

	aspectR := float32(h.swapChainExtend.Width) / float32(h.swapChainExtend.Height)
	ubo.proj.Perspective(45, aspectR, 0.1, 10)

	ubo.proj[1][1] *= -1

	vk.Memcopy(h.uniformBuffersMapped[currentImage], unsafer.StructToBytes(&ubo))
}

func (h *HelloTriangleApp) cleanup() error {
	return nil
}

type Vertex struct {
	pos   linmath.Vec2
	color linmath.Vec3
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

func GetVertexAttributeDescriptions() [2]vk.VertexInputAttributeDescription {
	attrDescr := [2]vk.VertexInputAttributeDescription{
		{
			Binding:  0,
			Location: 0,
			Format:   vk.FormatR32g32Sfloat,
			Offset:   uint32(unsafe.Offsetof(Vertex{}.pos)),
		},
		{
			Binding:  0,
			Location: 1,
			Format:   vk.FormatR32g32b32Sfloat,
			Offset:   uint32(unsafe.Offsetof(Vertex{}.color)),
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

func repackUint32(data []byte) []uint32 {
	buf := make([]uint32, len(data)/4)
	vk.Memcopy(unsafe.Pointer((*reflect.SliceHeader)(unsafe.Pointer(&buf)).Data), data)
	return buf
}
