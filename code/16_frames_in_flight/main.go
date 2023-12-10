package main

import (
	"cmp"
	"flag"
	"fmt"
	"log"
	"math"
	"runtime"

	"github.com/ironsmile/vulkan-tutorial-go/code/16_frames_in_flight/shaders"
	"github.com/ironsmile/vulkan-tutorial-go/code/optional"
	"github.com/ironsmile/vulkan-tutorial-go/code/unsafer"

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

const (
	title             = "Vulkan Tutorial: Frames in flight"
	maxFramesInFlight = 2
)

func main() {
	flag.Parse()

	app := &VulkanTutorialApp{
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

// VulkanTutorialApp is the first program from vulkan-tutorial.com
type VulkanTutorialApp struct {
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

	commandPool    vk.CommandPool
	commandBuffers []vk.CommandBuffer

	imageAvailabmeSems []vk.Semaphore
	renderFinishedSems []vk.Semaphore
	inFlightFences     []vk.Fence

	curentFrame uint32
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

	if err := a.cleanup(); err != nil {
		return fmt.Errorf("cleanup: %w", err)
	}

	return nil
}

func (a *VulkanTutorialApp) initWindow() error {
	if err := glfw.Init(); err != nil {
		return fmt.Errorf("glfw.Init: %w", err)
	}

	glfw.WindowHint(glfw.ClientAPI, glfw.NoAPI)
	glfw.WindowHint(glfw.Resizable, glfw.False)

	window, err := glfw.CreateWindow(a.width, a.height, title, nil, nil)
	if err != nil {
		return fmt.Errorf("creating window: %w", err)
	}

	a.window = window
	return nil
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

	if err := a.createGraphicsPipeline(); err != nil {
		return fmt.Errorf("createGraphicsPipeline: %w", err)
	}

	if err := a.createFramebuffers(); err != nil {
		return fmt.Errorf("createFramebuffers: %w", err)
	}

	if err := a.createCommandPool(); err != nil {
		return fmt.Errorf("createCommandPool: %w", err)
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

	for _, frameBuffer := range a.swapChainFramebuffers {
		vk.DestroyFramebuffer(a.device, frameBuffer, nil)
	}

	vk.DestroyRenderPass(a.device, a.renderPass, nil)

	for _, imageView := range a.swapChainImageViews {
		vk.DestroyImageView(a.device, imageView, nil)
	}

	if a.swapChain != vk.NullSwapchain {
		vk.DestroySwapchain(a.device, a.swapChain, nil)
	}
	a.swapChainImages = nil
	a.swapChainImageViews = nil

	if a.device != vk.Device(vk.NullHandle) {
		vk.DestroyDevice(a.device, nil)
	}
	if a.surface != vk.NullSurface {
		vk.DestroySurface(a.instance, a.surface, nil)
	}
	vk.DestroyInstance(a.instance, nil)
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

	//!TODO: left for later use
	deviceFeatures := []vk.PhysicalDeviceFeatures{{}}

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

func (a *VulkanTutorialApp) createImageViews() error {
	for i, swapChainImage := range a.swapChainImages {
		swapChainImage := swapChainImage
		createInfo := vk.ImageViewCreateInfo{
			SType:    vk.StructureTypeImageViewCreateInfo,
			Image:    swapChainImage,
			ViewType: vk.ImageViewType2d,
			Format:   a.swapChainImageFormat,
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
		res := vk.CreateImageView(a.device, &createInfo, nil, &imageView)
		if err := vk.Error(res); err != nil {
			return fmt.Errorf("failed to create image %d: %w", i, err)
		}

		a.swapChainImageViews = append(a.swapChainImageViews, imageView)
	}

	return nil
}

func (a *VulkanTutorialApp) createRenderPass() error {
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
	res := vk.CreateRenderPass(a.device, &rederPassInfo, nil, &renderPass)
	if err := vk.Error(res); err != nil {
		return fmt.Errorf("failed to create render pass: %w", err)
	}
	a.renderPass = renderPass

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
	res := vk.CreatePipelineLayout(a.device, &pipelineLayoutInfo, nil, &pipelineLayout)
	if err := vk.Error(res); err != nil {
		return fmt.Errorf("failed to create pipeline layout: %w", err)
	}
	a.pipelineLayout = pipelineLayout

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
		}

		frameBufferInfo := vk.FramebufferCreateInfo{
			SType:           vk.StructureTypeFramebufferCreateInfo,
			RenderPass:      a.renderPass,
			AttachmentCount: 1,
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

	clearColor := vk.NewClearValue([]float32{0, 0, 0, 1})

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
		ClearValueCount: 1,
		PClearValues:    []vk.ClearValue{clearColor},
	}

	vk.CmdBeginRenderPass(commandBuffer, &renderPassInfo, vk.SubpassContentsInline)
	vk.CmdBindPipeline(commandBuffer, vk.PipelineBindPointGraphics, a.graphicsPipline)

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

	vk.CmdDraw(commandBuffer, 3, 1, 0, 0)
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

	return indices.IsComplete() && extensionsSupported && swapChainAdequate
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
	vk.ResetFences(a.device, 1, fences)

	var imageIndex uint32
	vk.AcquireNextImage(
		a.device,
		a.swapChain,
		math.MaxUint64,
		a.imageAvailabmeSems[a.curentFrame],
		vk.Fence(vk.NullHandle),
		&imageIndex,
	)

	commandBuffer := a.commandBuffers[a.curentFrame]

	vk.ResetCommandBuffer(commandBuffer, 0)
	if err := a.recordCommandBuffer(commandBuffer, imageIndex); err != nil {
		return fmt.Errorf("recording command buffer: %w", err)
	}

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

	res := vk.QueueSubmit(
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

	vk.QueuePresent(a.presentQueue, &presentInfo)

	a.curentFrame = (a.curentFrame + 1) % maxFramesInFlight
	return nil
}

func (a *VulkanTutorialApp) cleanup() error {
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
