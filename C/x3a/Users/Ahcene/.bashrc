# Lancer l'agent SSH automatiquement
eval "$(ssh-agent -s)" > /dev/null

# Ajouter automatiquement la clé privée
if [ -f /c/Users/Ahcene/.ssh/id_ed25519 ]; then
  ssh-add /c/Users/Ahcene/.ssh/id_ed25519 2>/dev/null
fi


