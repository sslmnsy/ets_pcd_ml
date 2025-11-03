extends Control

func _ready():
	# Hubungkan semua tombol
	$MenuContainer/StartButton.pressed.connect(_on_start_pressed)
	$MenuContainer/GuideButton.pressed.connect(_on_guide_pressed)
	$MenuContainer/AboutButton.pressed.connect(_on_about_pressed)
	$MenuContainer/QuitButton.pressed.connect(_on_quit_pressed)

func _on_start_pressed():
	get_tree().change_scene_to_file("res://webcam_ui.tscn")

func _on_guide_pressed():
	get_tree().change_scene_to_file("res://guide.tscn")

func _on_about_pressed():
	get_tree().change_scene_to_file("res://about_team.tscn")

func _on_quit_pressed():
	get_tree().quit()
