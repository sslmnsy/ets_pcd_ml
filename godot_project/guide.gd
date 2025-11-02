extends Control

func _ready():
	# Hubungkan tombol kembali
	$BackButton.pressed.connect(_on_back_pressed)

func _on_back_pressed():
	# Kembali ke halaman utama
	get_tree().change_scene_to_file("res://main_menu.tscn")
