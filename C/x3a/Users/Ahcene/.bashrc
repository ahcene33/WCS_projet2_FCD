# Lancer automatiquement ssh-agent et ajouter la clé privée
eval "$(ssh-agent -s)" > /dev/null

if [ -f /c/Users/Ahcene/.ssh/id_ed25519 ]; then
  ssh-add /c/Users/Ahcene/.ssh/id_ed25519 2>/dev/null
fi

