var hyperaudiolite = (function () {
  
  var hal = {},
    transcriptId,
    transcript,
    playerId,
    player,
    melarr = [];

  player = document.getElementById("myAudio");
  
  function init() {
    var mels = document.querySelectorAll('[data-m]');
    for (var i = 0; i < mels.length; ++i) {
      var m = parseInt(mels[i].getAttribute('data-m'));
      var p = mels[i].parentNode;
      while (p !== document) {
        if (p.tagName.toLowerCase() === 'p' || p.tagName.toLowerCase() === 'figure' || p.tagName.toLowerCase() === 'ul') {
          break;
        };
        p = p.parentNode;
      };
      melarr[i] = { 'n': mels[i], 'm': m, 'p': p }
    };

    melarr.sort(function(a, b) { return a['m'] - b['m']; });

    for (var i = 0; i < melarr.length; ++i) {
      melarr[i].n.className = "unread";
    };
  };

  function setPlayHead(e) {
    var datam = parseInt(e.target.getAttribute("data-m"));

    if (!isNaN(datam)) {
      player.currentTime = datam / 1000;
      player.play();
    };
  };

  function checkPlayHead(e) {
    // binary search via http://stackoverflow.com/a/14370245
    var l = 0, r = melarr.length - 1;
    while (l <= r) {
      var m = l + ((r - l) >> 1);
      var comp = melarr[m].m / 1000 - player.currentTime;
      if (comp < 0) // arr[m] comes before the element
        l = m + 1;
      else if (comp > 0) // arr[m] comes after the element
        r = m - 1;
      else { // this[m] equals the element
        l = m;
        break;
      };
    };

    for (var i = 0; i < l; ++i) {
      melarr[i].n.className = "read";
    };
    for (var i = l; i < melarr.length; ++i) {
      melarr[i].n.className = "unread";
    };
  };

  hal.init = function(transcriptId, mediaElementId) {
    transcriptId = transcriptId;
    transcript = document.getElementById(transcriptId);
    playerId = mediaElementId;
    player = document.getElementById(playerId);
    init();
    player.addEventListener("timeupdate", checkPlayHead, false);
    transcript.addEventListener("click", setPlayHead, false);
  };
 
  return hal;
 
})();
