var hyperaudiocontrols = (function () {
  
  var hac = {},
    transcriptId,
    transcript,
    playerId,
    player,
    resetplay,
    resetpause,
    playanimId,
    playanim,
    pauseanimId,
    pauseanim;
  
  // https://gist.github.com/mrdoob/838785
  if (!window.requestAnimationFrame) {
    window.requestAnimationFrame = (function() {
      return window.webkitRequestAnimationFrame || window.mozRequestAnimationFrame ||
      window.oRequestAnimationFrame || window.msRequestAnimationFrame ||
      function( /* function FrameRequestCallback */ callback, /* DOMElement Element */ element ) {
        window.setTimeout( callback, 1000 / 60 );
      };
    })();
  };

  function togglePlayer(e) {
    var ui = null;
    if (e.target.getAttribute('data-m') === null && e.target.id !== playerId && e.target.tagName.toLowerCase() !== 'a') {
      if (player.paused) {
        ui = playanim;
        player.play();
      } else {
        ui = pauseanim;
        player.pause();
      };
    } else if (e.target.id !== playerId && e.target.tagName.toLowerCase() !== 'a') {
      ui = playanim;
    };
    if (ui !== null) {
        var pagex = e.pageX;
        var pagey = e.pageY;
        if (pagex === undefined) {
        var rect = canvas.getBoundingClientRect();
          var pagex = e.clientX - rect.left;
          var pagey = e.clientY - rect.top;
        };
    
        window.requestAnimationFrame(function() {
        ui.className = 'icondiv';
        ui.style.left = pagex + 'px';
        ui.style.top = pagey + 'px';
        ui.style.display = '';
        window.requestAnimationFrame(function() {
          ui.className = 'icondiv icongrow';
          if (ui.id === playanimId) {
            if (resetplay) {
              clearTimeout(resetplay);
            };
            resetplay = setTimeout(function() {
              ui.className = 'icondiv';
              ui.style.display = 'none';
            }, 510);
          };
          if (ui.id === pauseanimId) {
            if (resetpause) {
              clearTimeout(resetpause);
            };
            resetpause = setTimeout(function() {
              ui.className = 'icondiv';
              ui.style.display = 'none';
            }, 510);
          };
        });
        });
    };
  };

  function playCursor() {
    document.documentElement.className = "play";
  };

  function pauseCursor() {
    document.documentElement.className = "pause";
  };

  hac.init = function(transcriptId, mediaElementId, playId, pauseId) {
    transcriptId = transcriptId;
    transcript = document.getElementById(transcriptId);
    playerId = mediaElementId;
    player = document.getElementById(playerId);
    playanimId = playId;
    playanim = document.getElementById(playanimId);
    pauseanimId = pauseId;
    pauseanim = document.getElementById(pauseanimId);
    document.documentElement.addEventListener("click", togglePlayer, false);
    player.addEventListener("canplay", playCursor, false);
    player.addEventListener("pause", playCursor, false);
    player.addEventListener("playing", pauseCursor, false);
  };
 
  return hac;
 
})();
