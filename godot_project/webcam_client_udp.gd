extends Control

@onready var texture_rect: TextureRect = $MainContainer/VideoContainer/VideoContent/AspectRatioContainer/TextureRect
@onready var status_label: Label = $MainContainer/Header/HeaderContent/MarginRight/StatusLabel
@onready var connect_button: Button = $MainContainer/ControlPanel/ControlContent/ButtonContainer/ConnectButton
@onready var quit_button: Button = $MainContainer/ControlPanel/ControlContent/ButtonContainer/QuitButton
@onready var no_signal_label: Label = $MainContainer/VideoContainer/VideoContent/AspectRatioContainer/NoSignalLabel
@onready var fps_label: Label = $MainContainer/InfoPanel/InfoContent/InfoGrid/FPSLabel
@onready var resolution_label: Label = $MainContainer/InfoPanel/InfoContent/InfoGrid/ResolutionLabel
@onready var data_rate_label: Label = $MainContainer/InfoPanel/InfoContent/InfoGrid/DataRateLabel

# Hat category buttons
@onready var hat_buttons = {
	#"ASCOT CAP": $MainContainer/HatControlPanel/HatContent/HatContainer/HatGrid/AscotCapBtn,
	#"BASEBALL CAP": $MainContainer/HatControlPanel/HatContent/HatContainer/HatGrid/BaseballCapBtn,
	#"BERET": $MainContainer/HatControlPanel/HatContent/HatContainer/HatGrid/BeretBtn,
	#"BICORNE": $MainContainer/HatControlPanel/HatContent/HatContainer/HatGrid/BicorneBtn,
	#"BOATER": $MainContainer/HatControlPanel/HatContent/HatContainer/HatGrid/BoaterBtn,
	# "BOWLER": $MainContainer/HatControlPanel/HatContent/HatContainer/HatGrid/BowlerBtn,
	# "DEERSTALKER": $MainContainer/HatControlPanel/HatContent/HatContainer/HatGrid/DeerstalkerBtn,
	 "FEDORA": $MainContainer/HatControlPanel/HatContent/HatContainer/HatGrid/FedoraBtn,
	# "FEZ": $MainContainer/HatControlPanel/HatContent/HatContainer/HatGrid/FezBtn,
	# "FOOTBALL HELMET": $MainContainer/HatControlPanel/HatContent/HatContainer/HatGrid/FootballHelmetBtn,
	# "GARRISON CAP": $MainContainer/HatControlPanel/HatContent/HatContainer/HatGrid/GarrisonCapBtn,
	# "HARD HAT": $MainContainer/HatControlPanel/HatContent/HatContainer/HatGrid/HardHatBtn,
	# "MILITARY HELMET": $MainContainer/HatControlPanel/HatContent/HatContainer/HatGrid/MilitaryHelmetBtn,
	# "MOTARBOARD": $MainContainer/HatControlPanel/HatContent/HatContainer/HatGrid/MotarboardBtn,
	"PITH HELMET": $MainContainer/HatControlPanel/HatContent/HatContainer/HatGrid/PithHelmetBtn,
	"PORK PIE": $MainContainer/HatControlPanel/HatContent/HatContainer/HatGrid/PorkPieBtn,
	"SOMBRERO": $MainContainer/HatControlPanel/HatContent/HatContainer/HatGrid/SombreroBtn,
	# "SOUTHWESTER": $MainContainer/HatControlPanel/HatContent/HatContainer/HatGrid/SouthwesterBtn,
	"TOP HAT": $MainContainer/HatControlPanel/HatContent/HatContainer/HatGrid/TopHatBtn,
	# "ZUCCHETTO": $MainContainer/HatControlPanel/HatContent/HatContainer/HatGrid/ZucchettoBtn
}

var udp_client: PacketPeerUDP
var is_connected: bool = false
var server_host: String = "127.0.0.1"
var server_port: int = 8888

# Hat control
var current_hat_category: String = ""

# Frame reassembly
var frame_buffers: Dictionary = {}  # seq_num -> {total_packets, received_packets, data_parts}
var last_completed_sequence: int = 0
var frame_timeout: float = 1.0  # 1 detik timeout untuk frame

# Performance monitoring
var frame_count: int = 0
var last_fps_time: float = 0.0
var current_fps: float = 0.0
var bytes_received: int = 0
var last_data_rate_time: float = 0.0
var current_data_rate: float = 0.0

# Packet statistics
var packets_received: int = 0
var frames_completed: int = 0
var frames_dropped: int = 0

func _ready():
	udp_client = PacketPeerUDP.new()
	connect_button.pressed.connect(_on_connect_button_pressed)
	quit_button.pressed.connect(_on_quit_button_pressed)
	
	for category in hat_buttons:
		hat_buttons[category].pressed.connect(_on_hat_category_selected.bind(category))
	
	update_status("Ready to connect")
	update_info_display()
	no_signal_label.visible = true
	
	print("🎮 Godot UDP client initialized")
	print("Target server: ", server_host, ":", server_port)
	print("🎩 Hat categories loaded: ", hat_buttons.size())

func _on_quit_button_pressed():
	get_tree().change_scene_to_file("res://main_menu.tscn")

func _process(delta):
	if is_connected:
		receive_packets()
		cleanup_old_frames()
		update_performance_metrics(delta)
	
	# MODIFIED: Kontrol Panah Kiri/Kanan DIHAPUS
	# if is_connected:
	# 	if Input.is_action_just_pressed("ui_right"):
	# 		send_hat_command("NEXT_HAT")
	# 	if Input.is_action_just_pressed("ui_left"):
	# 		send_hat_command("PREV_HAT")

func _on_connect_button_pressed():
	if is_connected:
		disconnect_from_server()
	else:
		connect_to_server()

func send_hat_command(command: String):
	"""Kirim pesan teks sederhana ke server."""
	if not is_connected:
		print("⚠️ Not connected, cannot send command: ", command)
		return
		
	var message = command.to_utf8_buffer()
	var send_result = udp_client.put_packet(message)
	
	if send_result == OK:
		print("📤 Perintah terkirim: ", command)
	else:
		print("❌ Gagal mengirim perintah: ", command, " Error: ", send_result)

func _on_hat_category_selected(category: String):
	print("🎩 Tombol Kategori Ditekan: ", category)
	
	# MODIFIED: Logika Toggle
	if category == current_hat_category:
		# Jika menekan tombol yang sama, matikan topi
		print("Mematikan topi (toggle).")
		send_hat_command("HAT_OFF")
		current_hat_category = "" # Reset
	else:
		# Jika menekan tombol baru, ganti topi
		print("Mengganti topi ke: ", category)
		send_hat_command("HAT_CATEGORY:" + category)
		current_hat_category = category # Simpan status


func connect_to_server():
	if is_connected:
		print("⚠️  Already connected!")
		return
	
	print("🔄 Starting UDP connection...")
	update_status("Connecting...")
	
	var error = udp_client.connect_to_host(server_host, server_port)
	
	if error != OK:
		update_status("Failed to setup UDP - Error: " + str(error))
		print("❌ UDP setup failed: ", error)
		return
	
	var registration_message = "REGISTER".to_utf8_buffer()
	var send_result = udp_client.put_packet(registration_message)
	
	if send_result != OK:
		update_status("Failed to register - Error: " + str(send_result))
		print("❌ Registration failed: ", send_result)
		return
	
	print("📤 Registration sent, waiting for confirmation...")
	
	var timeout = 0
	var max_timeout = 180
	var confirmed = false
	
	while timeout < max_timeout and not confirmed:
		await get_tree().process_frame
		timeout += 1
		
		if udp_client.get_available_packet_count() > 0:
			var packet = udp_client.get_packet()
			var message = packet.get_string_from_utf8()
			
			if message == "REGISTERED":
				confirmed = true
				print("✅ Registration confirmed!")
			elif message == "SERVER_SHUTDOWN":
				update_status("Server is shutting down")
				return
	
	if confirmed:
		is_connected = true
		update_status("Connected - Receiving video...")
		connect_button.text = "Disconnect"
		print("🎥 Ready to receive video streams!")
		
		packets_received = 0
		frames_completed = 0
		frames_dropped = 0
		frame_buffers.clear()
		current_hat_category = "" # NEW: Reset status topi
	else:
		update_status("Registration timeout")
		print("❌ Registration timeout")
		udp_client.close()

func disconnect_from_server():
	print("🔌 Disconnecting from server...")
	
	if is_connected:
		var unregister_message = "UNREGISTER".to_utf8_buffer()
		udp_client.put_packet(unregister_message)
	
	is_connected = false
	udp_client.close()
	frame_buffers.clear()
	
	update_status("Disconnected")
	connect_button.text = "Connect to Server"
	
	texture_rect.texture = null
	no_signal_label.visible = true
	
	frame_count = 0
	bytes_received = 0
	current_fps = 0.0
	current_data_rate = 0.0
	current_hat_category = "" # NEW: Reset status topi
	update_info_display()

func receive_packets():
	var packet_count = udp_client.get_available_packet_count()
	
	for i in range(packet_count):
		var packet = udp_client.get_packet()
		if packet.size() >= 12:
			packets_received += 1
			bytes_received += packet.size()
			process_packet(packet)

func process_packet(packet: PackedByteArray):
	if packet.size() < 12:
		return
	
	var sequence_number = bytes_to_int(packet.slice(0, 4))
	var total_packets = bytes_to_int(packet.slice(4, 8))
	var packet_index = bytes_to_int(packet.slice(8, 12))
	var packet_data = packet.slice(12)
	
	if total_packets <= 0 or packet_index >= total_packets or sequence_number <= 0:
		print("⚠️  Invalid packet header: seq=", sequence_number, " total=", total_packets, " index=", packet_index)
		return
	
	if sequence_number < last_completed_sequence - 2:
		return
	
	if sequence_number not in frame_buffers:
		frame_buffers[sequence_number] = {
			"total_packets": total_packets,
			"received_packets": 0,
			"data_parts": {},
			"timestamp": Time.get_ticks_msec() / 1000.0
		}
	
	var frame_buffer = frame_buffers[sequence_number]
	
	if packet_index not in frame_buffer.data_parts:
		frame_buffer.data_parts[packet_index] = packet_data
		frame_buffer.received_packets += 1
		
		if frame_buffer.received_packets == frame_buffer.total_packets:
			assemble_and_display_frame(sequence_number)

func assemble_and_display_frame(sequence_number: int):
	if sequence_number not in frame_buffers:
		return
	
	var frame_buffer = frame_buffers[sequence_number]
	var frame_data = PackedByteArray()
	
	for i in range(frame_buffer.total_packets):
		if i in frame_buffer.data_parts:
			frame_data.append_array(frame_buffer.data_parts[i])
		else:
			print("❌ Missing packet ", i, " for frame ", sequence_number)
			frames_dropped += 1
			frame_buffers.erase(sequence_number)
			return
	
	frame_buffers.erase(sequence_number)
	last_completed_sequence = sequence_number
	frames_completed += 1
	
	display_frame(frame_data)
	
	if frames_completed % 30 == 0:
		var drop_rate = 0.0
		if (frames_completed + frames_dropped) > 0:
			drop_rate = float(frames_dropped) / float(frames_completed + frames_dropped) * 100.0
		print("📊 Frame ", sequence_number, " completed. Drop rate: %.1f%%" % drop_rate)

func cleanup_old_frames():
	var current_time = Time.get_ticks_msec() / 1000.0
	var sequences_to_remove = []
	
	for seq_num in frame_buffers:
		var frame_buffer = frame_buffers[seq_num]
		if current_time - frame_buffer.timestamp > frame_timeout:
			sequences_to_remove.append(seq_num)
			frames_dropped += 1
	
	for seq_num in sequences_to_remove:
		frame_buffers.erase(seq_num)
		if sequences_to_remove.size() > 0:
			print("🗑️  Cleaned up ", sequences_to_remove.size(), " timed out frames")

func bytes_to_int(bytes: PackedByteArray) -> int:
	if bytes.size() != 4:
		return 0
	
	return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3]

func display_frame(frame_data: PackedByteArray):
	var image = Image.new()
	var error = image.load_jpg_from_buffer(frame_data)
	
	if error == OK:
		var texture = ImageTexture.new()
		texture.set_image(image)
		
		texture_rect.texture = texture
		no_signal_label.visible = false
		
		resolution_label.text = "Resolution: %dx%d" % [image.get_width(), image.get_height()]
		
		frame_count += 1
	else:
		print("❌ Error loading image: ", error)

func update_performance_metrics(delta: float):
	last_fps_time += delta
	if last_fps_time >= 1.0:
		current_fps = frame_count / last_fps_time
		frame_count = 0
		last_fps_time = 0.0
	
	last_data_rate_time += delta
	if last_data_rate_time >= 1.0:
		current_data_rate = bytes_received / last_data_rate_time / 1024.0
		bytes_received = 0
		last_data_rate_time = 0.0
		
	update_info_display()

func update_info_display():
	if is_connected:
		fps_label.text = "FPS: %.1f" % current_fps
		data_rate_label.text = "Data Rate: %.1f KB/s" % current_data_rate
		
		if frames_completed + frames_dropped > 0:
			var drop_rate = float(frames_dropped) / float(frames_completed + frames_dropped) * 100.0
			status_label.text = "Status: Connected - Packets: %d, Drop: %.1f%%" % [packets_received, drop_rate]
	else:
		fps_label.text = "FPS: --"
		resolution_label.text = "Resolution: --"
		data_rate_label.text = "Data Rate: --"

func update_status(message: String):
	status_label.text = "Status: " + message
	print("🎮 Webcam Client: " + message) 

func _notification(what):
	if what == NOTIFICATION_WM_CLOSE_REQUEST:
		if is_connected:
			disconnect_from_server()
		get_tree().change_scene_to_file("res://main_menu.tscn")
