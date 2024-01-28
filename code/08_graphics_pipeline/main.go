package main

import (
	"cmp"
	"flag"
	"fmt"
	"log"
	"math"
	"runtime"

	"github.com/go-gl/glfw/v3.3/glfw"
	"github.com/ironsmile/vulkan-tutorial-go/code/optional"
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
	title = "Vulkan Tutorial: Graphics Pipeline - Introduction"
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

	if err := a.createInstance(); err != nil {
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

	if err := a.createGraphicsPipeline(); err != nil {
		return fmt.Errorf("createGraphicsPipeline: %w", err)
	}

	return nil
}

func (a *VulkanTutorialApp) cleanupVulkan() {
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

func (a *VulkanTutorialApp) createGraphicsPipeline() error {
	return nil
}

func (a *VulkanTutorialApp) createInstance() error {
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
	if res := vk.CreateInstance(&createInfo, nil, &instance); res != vk.Success {
		return fmt.Errorf("failed to create Vulkan instance: %w", vk.Error(res))
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

func (a *VulkanTutorialApp) mainLoop() error {
	log.Printf("main loop!\n")

	for !a.window.ShouldClose() {
		glfw.PollEvents()
	}

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
