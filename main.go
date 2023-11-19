package main

import (
	"cmp"
	"flag"
	"fmt"
	"log"
	"math"
	"runtime"
	"unsafe"

	"vulkan-tutorial/queues"
	"vulkan-tutorial/shaders"

	"github.com/go-gl/glfw/v3.3/glfw"
	vk "github.com/vulkan-go/vulkan"
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

func main() {
	flag.Parse()

	app := &HelloTriangleApp{
		width:                  1024,
		height:                 768,
		enableValidationLayers: args.debug,
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

	graphicsQueue vk.Queue
	presentQueue  vk.Queue

	surface vk.Surface

	swapChain            vk.Swapchain
	swapChainImages      []vk.Image
	swapChainImageViews  []vk.ImageView
	swapChainImageFormat vk.Format
	swapChainExtend      vk.Extent2D

	swapChainFramebuffers []vk.Framebuffer

	renderPass     vk.RenderPass
	pipelineLayout vk.PipelineLayout

	graphicsPipline vk.Pipeline

	commandPool   vk.CommandPool
	commandBuffer vk.CommandBuffer

	imageAvailabmeSem vk.Semaphore
	renderFinishedSem vk.Semaphore
	inFlightFence     vk.Fence
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
	defer h.cleanVulkan()

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
	glfw.WindowHint(glfw.Resizable, glfw.False)

	window, err := glfw.CreateWindow(h.width, h.height, title, nil, nil)
	if err != nil {
		return fmt.Errorf("creating window: %w", err)
	}

	h.window = window
	return nil
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

	if err := h.createGraphicsPipeline(); err != nil {
		return fmt.Errorf("createGraphicsPipeline: %w", err)
	}

	if err := h.createFramebuffers(); err != nil {
		return fmt.Errorf("createFramebuffers: %w", err)
	}

	if err := h.createCommandPool(); err != nil {
		return fmt.Errorf("createCommandPool: %w", err)
	}

	if err := h.createCommandBuffer(); err != nil {
		return fmt.Errorf("createCommandBuffer: %w", err)
	}

	if err := h.createSyncObjects(); err != nil {
		return fmt.Errorf("createSyncObjects: %w", err)
	}

	return nil
}

func (h *HelloTriangleApp) cleanVulkan() {
	vk.DestroySemaphore(h.device, h.imageAvailabmeSem, nil)
	vk.DestroySemaphore(h.device, h.renderFinishedSem, nil)
	vk.DestroyFence(h.device, h.inFlightFence, nil)

	vk.DestroyCommandPool(h.device, h.commandPool, nil)

	vk.DestroyPipeline(h.device, h.graphicsPipline, nil)
	vk.DestroyPipelineLayout(h.device, h.pipelineLayout, nil)

	for _, frameBuffer := range h.swapChainFramebuffers {
		vk.DestroyFramebuffer(h.device, frameBuffer, nil)
	}

	vk.DestroyRenderPass(h.device, h.renderPass, nil)

	for _, imageView := range h.swapChainImageViews {
		vk.DestroyImageView(h.device, imageView, nil)
	}

	if h.swapChain != vk.NullSwapchain {
		vk.DestroySwapchain(h.device, h.swapChain, nil)
	}
	h.swapChainImages = nil
	h.swapChainImageViews = nil

	if h.device != vk.Device(vk.NullHandle) {
		vk.DestroyDevice(h.device, nil)
	}
	if h.surface != vk.NullSurface {
		vk.DestroySurface(h.instance, h.surface, nil)
	}
	vk.DestroyInstance(h.instance, nil)
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

	vertexInputInfo := vk.PipelineVertexInputStateCreateInfo{
		SType: vk.StructureTypePipelineVertexInputStateCreateInfo,

		VertexBindingDescriptionCount:   0,
		VertexAttributeDescriptionCount: 0,
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
		FrontFace:               vk.FrontFaceClockwise,
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
		SetLayoutCount: 0,
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

func (h *HelloTriangleApp) createCommandBuffer() error {
	allocInfo := vk.CommandBufferAllocateInfo{
		SType:              vk.StructureTypeCommandBufferAllocateInfo,
		CommandPool:        h.commandPool,
		Level:              vk.CommandBufferLevelPrimary,
		CommandBufferCount: 1,
	}

	commandBuffers := make([]vk.CommandBuffer, 1)
	res := vk.AllocateCommandBuffers(h.device, &allocInfo, commandBuffers)
	if err := vk.Error(res); err != nil {
		return fmt.Errorf("failed to allocate command buffer: %w", err)
	}
	h.commandBuffer = commandBuffers[0]

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

	res := vk.BeginCommandBuffer(h.commandBuffer, &beginInfo)
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

	vk.CmdDraw(commandBuffer, 3, 1, 0, 0)
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

	var imageAvailabmeSem vk.Semaphore
	if err := vk.Error(
		vk.CreateSemaphore(h.device, &semaphoreInfo, nil, &imageAvailabmeSem),
	); err != nil {
		return fmt.Errorf("failed to create imageAvailabmeSem: %w", err)
	}
	h.imageAvailabmeSem = imageAvailabmeSem

	var renderFinishedSem vk.Semaphore
	if err := vk.Error(
		vk.CreateSemaphore(h.device, &semaphoreInfo, nil, &renderFinishedSem),
	); err != nil {
		return fmt.Errorf("failed to create renderFinishedSem: %w", err)
	}
	h.renderFinishedSem = renderFinishedSem

	var fence vk.Fence
	if err := vk.Error(
		vk.CreateFence(h.device, &fenceInfo, nil, &fence),
	); err != nil {
		return fmt.Errorf("failed to create inFlightFence: %w", err)
	}
	h.inFlightFence = fence

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
	vk.WaitForFences(h.device, 1, []vk.Fence{h.inFlightFence}, vk.True, math.MaxUint64)
	vk.ResetFences(h.device, 1, []vk.Fence{h.inFlightFence})

	var imageIndex uint32
	vk.AcquireNextImage(
		h.device,
		h.swapChain,
		math.MaxUint64,
		h.imageAvailabmeSem,
		vk.Fence(vk.NullHandle),
		&imageIndex,
	)

	vk.ResetCommandBuffer(h.commandBuffer, 0)
	if err := h.recordCommandBuffer(h.commandBuffer, imageIndex); err != nil {
		return fmt.Errorf("recording command buffer: %w", err)
	}

	signalSemaphores := []vk.Semaphore{
		h.renderFinishedSem,
	}

	submitInfo := vk.SubmitInfo{
		SType:              vk.StructureTypeSubmitInfo,
		WaitSemaphoreCount: 1,
		PWaitSemaphores:    []vk.Semaphore{h.imageAvailabmeSem},
		PWaitDstStageMask: []vk.PipelineStageFlags{
			vk.PipelineStageFlags(vk.PipelineStageColorAttachmentOutputBit),
		},
		CommandBufferCount:   1,
		PCommandBuffers:      []vk.CommandBuffer{h.commandBuffer},
		PSignalSemaphores:    signalSemaphores,
		SignalSemaphoreCount: uint32(len(signalSemaphores)),
	}

	res := vk.QueueSubmit(
		h.graphicsQueue,
		1,
		[]vk.SubmitInfo{submitInfo},
		h.inFlightFence,
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

	vk.QueuePresent(h.presentQueue, &presentInfo)
	return nil
}

func (h *HelloTriangleApp) cleanup() error {
	return nil
}

// swapChainSupportDetails describes a present surface. The type is suitable for
// passing around many details of the service between functions.
type swapChainSupportDetails struct {
	capabilities vk.SurfaceCapabilities
	formats      []vk.SurfaceFormat
	presentModes []vk.PresentMode
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
	vk.Memcopy(unsafe.Pointer((*sliceHeader)(unsafe.Pointer(&buf)).Data), data)
	return buf
}

type sliceHeader struct {
	Data uintptr
	Len  int
	Cap  int
}

const title = "Vulkan Tutorial: Hello Triangle"
